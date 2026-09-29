"""Generate three-condition samples from a train_triple.py checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from labels import parse_condition
from models.triple import TripleDDIMSampler, TripleUNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    style, color, sex = parse_condition(args.condition)
    if args.count < 1 or args.batch_size < 1:
        parser.error("count and batch-size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["args"]
    if args.condition != config["held_out"]:
        parser.error("Requested condition differs from the checkpoint's held-out condition")
    model = TripleUNet(compose=config["compose"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    sampler = TripleDDIMSampler(model).cuda().eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for start in range(0, args.count, args.batch_size):
        n = min(args.batch_size, args.count - start)
        x = torch.randn((n, 3, 64, 64), generator=generator, device="cuda")
        labels = [torch.full((n,), value, dtype=torch.long, device="cuda")
                  for value in (style, color, sex)]
        images = sampler(x, *labels)
        for i, image in enumerate(images):
            save_image(image.clamp(-1, 1).add(1).div(2), out / f"{start+i:05d}.png")
    (out / "generation.json").write_text(json.dumps({"checkpoint": args.checkpoint,
        "condition": args.condition, "count": args.count, "seed": args.seed,
        "guidance_w": 1.8, "ddim_steps": 100, "eta": 0}, indent=2) + "\n")


if __name__ == "__main__":
    main()
