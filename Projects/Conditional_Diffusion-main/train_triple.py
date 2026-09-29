"""Train one held-out three-condition CelebA CCDM model.

The data directory contains JPGs in folders such as
`Wavy_Hair Black_Hair Female/`. No generated outcomes are used.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from labels_triple import COLOR, SEX, STYLE, parse_condition
from models.triple import TripleDiffusionTrainer, TripleUNet


class CelebATriples(Dataset):
    def __init__(self, root, held_out):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        self.rows = []
        for folder in sorted(Path(root).iterdir()):
            if not folder.is_dir():
                continue
            condition = parse_condition(folder.name)
            if folder.name != held_out:
                self.rows.extend((p, condition) for p in sorted(folder.glob("*.jpg")))
        if not self.rows:
            raise ValueError("No training JPGs found outside the held-out condition")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, (style, color, sex) = self.rows[index]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, style, color, sex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CelebA condition-folder root")
    parser.add_argument("--held-out", required=True, help="e.g. 'Wavy_Hair Black_Hair Female'")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--compose", action="store_true", help="learn projection of three embeddings")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    parse_condition(args.held_out)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = CelebATriples(args.data, args.held_out)
    counts = Counter(condition for _, condition in dataset.rows)
    weights = torch.tensor([1.0 / counts[c] for _, c in dataset.rows], dtype=torch.double)
    balance = WeightedRandomSampler(weights, len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=balance,
                        num_workers=args.workers, pin_memory=True)
    model = TripleUNet(compose=args.compose).cuda()
    trainer = TripleDiffusionTrainer(model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 10))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                total_iters=min(10, args.epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[min(10, args.epochs)]) if args.epochs > 10 else warmup
    metadata = {"args": vars(args), "style": STYLE, "color": COLOR, "sex": SEX,
                "train_images": len(dataset), "tuple_counts": {str(k): v for k, v in counts.items()}}
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, style, color, sex in loader:
            x, style, color, sex = (z.cuda(non_blocking=True) for z in (x, style, color, sex))
            # Keep the source train.py loss scale for a matched implementation.
            loss = trainer(x, style, color, sex).sum() / x.shape[0] ** 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(json.dumps({"epoch": epoch, "mean_loss": total / len(loader),
                          "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        if epoch == args.epochs or epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": metadata},
                       out / f"model_{epoch}.pth")


if __name__ == "__main__":
    main()
