import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from dataset_test import CustomImageDatasetCondition_MPD
from models.engine import DDIMSamplerImage
from utils import get_model


def apply_crf(img, prob_map):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)
    h, w = prob_map.shape
    crf = dcrf.DenseCRF2D(w, h, 2)

    prob_map = np.clip(prob_map, 0.001, 0.999)
    denom = max(float(prob_map.max() - prob_map.min()), 1e-8)
    prob_map = (prob_map - prob_map.min()) / denom

    probs = np.zeros((2, h, w), dtype=np.float32)
    probs[1] = prob_map
    probs[0] = 1.0 - prob_map
    crf.setUnaryEnergy(unary_from_softmax(probs))
    crf.addPairwiseGaussian(sxy=2, compat=1)
    crf.addPairwiseBilateral(sxy=20, srgb=5, rgbim=img, compat=5)
    refined = np.argmax(crf.inference(3), axis=0).reshape((h, w))
    return refined.astype(np.float32)


def strip_module_prefix(state_dict):
    out = OrderedDict()
    for key, value in state_dict.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def mask_scores(pred, target, eps=1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)
    return iou, dice, precision, recall


def build_args(cli):
    class ModelArgs(argparse.Namespace):
        arch = "unetattention_image"
        img_size = 128
        num_timestep = 1000
        beta = (0.0001, 0.02)
        num_condition = [57, 1]
        emb_size = 128
        channel_mult = [1, 2, 2, 2]
        num_res_blocks = 2
        use_spatial_transformer = False
        num_heads = 4
        num_sample = 4
        w = 3
        projection_dim = 512
        only_table = False
        concat = False
        only_encoder = False
        num_head_channels = -1
        encoder_path = None
        steps = 100
        eval_batch_size = 8
        num_workers = 4
        ignored = None

    args = ModelArgs()
    args.img_size = cli.img_size
    args.steps = cli.steps
    args.eval_batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    return args


def checkpoint_path(cli):
    lr = f"{cli.ckpt_lr:.1e}"
    return os.path.join(
        cli.checkpoint_root,
        f"w_{cli.mpd_w:.1f}_lr{lr}",
        f"model_epoch{cli.epoch}_lr{lr}_w{cli.mpd_w:.1f}.pth",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/work/macw1030/isic_mpd_png")
    parser.add_argument("--checkpoint_root", default="checkpoint/MED_pth_w_ISIC")
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--ckpt_lr", type=float, default=5e-6)
    parser.add_argument("--mpd_w", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--largest_component", action="store_true")
    parser.add_argument("--fill_holes", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_csv", default=None)
    cli = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_args = build_args(cli)
    model = get_model(model_args)
    ckpt = torch.load(checkpoint_path(cli), map_location=device)["model"]
    model.load_state_dict(strip_module_prefix(ckpt))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    sampler = DDIMSamplerImage(
        model=model,
        beta=model_args.beta,
        T=model_args.num_timestep,
        w=model_args.w,
    ).to(device)
    sampler.eval()

    transform = transforms.Compose([
        transforms.Resize((cli.img_size, cli.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CustomImageDatasetCondition_MPD(
        data_root=cli.data_root,
        dataset_type="test",
        transform=transform,
    )
    if cli.max_samples is not None:
        dataset = Subset(dataset, list(range(min(cli.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
    )

    rows = []
    sums = np.zeros(4, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="generate-style eval")):
            c2_image = batch["input_image"].to(device)
            batch_size, channels, height, width = c2_image.shape
            x_i = torch.randn(
                cli.num_generations * batch_size,
                channels,
                height,
                width,
                device=device,
            )
            c2_repeat = c2_image.repeat(cli.num_generations, 1, 1, 1)
            x_i = (1.0 - cli.mpd_w) * x_i + cli.mpd_w * c2_repeat
            x0_all = sampler(x_i, steps=cli.steps)
            x0_prob = torch.sigmoid(x0_all.mean(dim=1, keepdim=True))

            processed = []
            for pred_idx in range(x0_prob.shape[0]):
                if cli.crf:
                    soft = x0_prob[pred_idx, 0].detach().cpu().numpy()
                    img = c2_repeat[pred_idx].permute(1, 2, 0).detach().cpu().numpy()
                    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    pred = apply_crf(img, soft)
                else:
                    pred = (x0_prob[pred_idx, 0].detach().cpu().numpy() > 0.5).astype(np.float32)

                if cli.largest_component:
                    labeled, num_features = ndi.label(pred)
                    sizes = ndi.sum(pred, labeled, range(num_features + 1))
                    if len(sizes) > 1:
                        largest_label = int(np.argmax(sizes[1:]) + 1)
                        pred = (labeled == largest_label).astype(np.float32)
                    else:
                        pred = np.zeros_like(pred, dtype=np.float32)
                if cli.fill_holes:
                    pred = binary_fill_holes(pred).astype(np.float32)
                processed.append(torch.tensor(pred, device=device))

            processed = torch.stack(processed).view(cli.num_generations, batch_size, height, width)
            final_masks = (processed.mean(dim=0) > 0.5).float()

            for idx in range(batch_size):
                sample_index = batch_idx * cli.batch_size + idx
                base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
                original_index = dataset.indices[sample_index] if isinstance(dataset, Subset) else sample_index
                original_image = Image.open(base_dataset.img_path_files[original_index]).convert("RGB")
                original_hw = (original_image.size[1], original_image.size[0])
                pred_resized = F.interpolate(
                    final_masks[idx].unsqueeze(0).unsqueeze(0),
                    size=original_hw,
                    mode="nearest",
                )[0, 0].detach().cpu().numpy()
                pred_np = pred_resized > 0.5
                gt_np = np.array(Image.open(base_dataset.gt_path_files[original_index]).convert("L")) > 127
                iou, dice, precision, recall = mask_scores(pred_np, gt_np)
                sums += np.array([iou, dice, precision, recall])
                count += 1
                rows.append({
                    "filename": batch["filename"][idx],
                    "iou": f"{iou:.6g}",
                    "dice": f"{dice:.6g}",
                    "precision": f"{precision:.6g}",
                    "recall": f"{recall:.6g}",
                })

            means = sums / max(1, count)
            print(
                f"progress {count}/{len(dataset)} "
                f"iou={means[0]:.4f} dice={means[1]:.4f} "
                f"precision={means[2]:.4f} recall={means[3]:.4f}",
                flush=True,
            )

    means = sums / max(1, count)
    print(
        f"FINAL epoch={cli.epoch} generations={cli.num_generations} "
        f"crf={cli.crf} largest_component={cli.largest_component} fill_holes={cli.fill_holes} "
        f"samples={count} iou={means[0]:.6f} dice={means[1]:.6f} "
        f"precision={means[2]:.6f} recall={means[3]:.6f}",
        flush=True,
    )

    if cli.output_csv:
        os.makedirs(os.path.dirname(cli.output_csv), exist_ok=True)
        with open(cli.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "iou", "dice", "precision", "recall"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
