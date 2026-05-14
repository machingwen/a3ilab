#!/usr/bin/env python3
"""Evaluate the original MPD reproduction checkpoint on ISIC with DenseCRF.

The two MPD weights are named explicitly:

* sampler_w: DDIM sampler/model guidance weight passed into DDIMSamplerImage.
* mpd_w: interpolation weight used to mix initial noise with the condition image.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from torch.utils.data import DataLoader
from torchvision import transforms

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_test import CustomImageDatasetCondition_MPD  # noqa: E402
from models.engine import DDIMSamplerImage  # noqa: E402
from utils import get_model  # noqa: E402


DEFAULT_DATA_ROOT = Path("/work-pvc/macw1030/isic_mpd_png_128")
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoint/isic_original_mpd_repro_128_e500/mpd_w_0.9_lr5.0e-06/"
    / "model_epoch500_lr5.0e-06_mpd_w0.9.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/isic_original_mpd_repro_128_e500/eval_densecrf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the original MPD reproduction checkpoint with DenseCRF post-processing."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pred_mask_dir",
        type=Path,
        default=None,
        help="Optional directory of masks created by generate_main.py. If set, evaluate these masks instead of running inference.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--sampler_w", type=float, default=3.0)
    parser.add_argument("--mpd_w", type=float, default=0.9)
    parser.add_argument("--save_masks", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_crf(img: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img, dtype=np.uint8)
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)

    height, width = prob_map.shape
    dense_crf = dcrf.DenseCRF2D(width, height, 2)

    prob_map = np.clip(prob_map, 0.001, 0.999)
    denom = prob_map.max() - prob_map.min()
    prob_map = (prob_map - prob_map.min()) / (denom if denom > 0 else 1.0)

    probs = np.zeros((2, height, width), dtype=np.float32)
    probs[1] = prob_map
    probs[0] = 1 - prob_map
    dense_crf.setUnaryEnergy(unary_from_softmax(probs))
    dense_crf.addPairwiseGaussian(sxy=2, compat=1)
    dense_crf.addPairwiseBilateral(sxy=20, srgb=5, rgbim=img, compat=5)

    return np.argmax(dense_crf.inference(3), axis=0).reshape((height, width))


def model_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        arch="unetattention_image",
        method="mpd",
        img_size=cli_args.img_size,
        num_timestep=1000,
        beta=(0.0001, 0.02),
        num_condition=[57, 1],
        emb_size=128,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        use_spatial_transformer=False,
        num_heads=4,
        num_sample=4,
        w=cli_args.sampler_w,
        projection_dim=512,
        only_table=False,
        concat=False,
        only_encoder=False,
        num_head_channels=-1,
        encoder_path=None,
        steps=cli_args.steps,
        eval_batch_size=cli_args.eval_batch_size,
        num_workers=cli_args.num_workers,
        ignored=None,
    )


def load_model_and_sampler(
    cli_args: argparse.Namespace,
    device: torch.device,
) -> tuple[argparse.Namespace, torch.nn.Module, torch.nn.Module]:
    args = model_args(cli_args)
    checkpoint = torch.load(cli_args.checkpoint, map_location=device)["model"]
    state_dict = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in checkpoint.items()
    )

    model = get_model(args)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    sampler = DDIMSamplerImage(
        model=model,
        beta=args.beta,
        T=args.num_timestep,
        w=args.w,
    ).to(device)
    sampler.eval()
    return args, model, sampler


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / (pred_pixels + gt_pixels) if pred_pixels + gt_pixels else 1.0
    return {
        "intersection": intersection,
        "union": union,
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
        "iou": float(iou),
        "dice": float(dice),
    }


def find_generated_mask(mask_dir: Path, filename: str, mpd_w: float) -> Path:
    candidates = [
        mask_dir / f"{filename}_mask_{mpd_w:g}.PNG",
        mask_dir / f"{filename}_mask_{mpd_w:.1f}.PNG",
        mask_dir / f"{filename}_mask_mpd_w{mpd_w:g}.PNG",
        mask_dir / f"{filename}_mask_mpd_w_{mpd_w:g}.PNG",
        mask_dir / f"{filename}_mask_mpd_w_{mpd_w:.1f}.PNG",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(mask_dir.glob(f"{filename}_mask_*.PNG"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No generated mask found for {filename} under {mask_dir}")


def evaluate_generated_masks(
    cli_args: argparse.Namespace,
    dataset: CustomImageDatasetCondition_MPD,
) -> dict[str, Any]:
    if cli_args.pred_mask_dir is None:
        raise ValueError("--pred_mask_dir is required for generated-mask evaluation")

    per_image_rows: list[dict[str, Any]] = []
    total_intersection = 0
    total_union = 0
    total_pred_pixels = 0
    total_gt_pixels = 0
    started_at = time.time()

    for idx, (_image_path, gt_path, filename) in enumerate(dataset.pairs):
        mask_path = find_generated_mask(cli_args.pred_mask_dir, filename, cli_args.mpd_w)
        pred = np.array(Image.open(mask_path).convert("L")) > 127
        gt = np.array(Image.open(gt_path).convert("L")) > 127
        if pred.shape != gt.shape:
            pred_image = Image.fromarray(pred.astype(np.uint8) * 255)
            nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
            pred = np.array(pred_image.resize((gt.shape[1], gt.shape[0]), nearest)) > 127

        metrics = compute_binary_metrics(pred, gt)
        total_intersection += int(metrics["intersection"])
        total_union += int(metrics["union"])
        total_pred_pixels += int(metrics["pred_pixels"])
        total_gt_pixels += int(metrics["gt_pixels"])

        per_image_rows.append(
            {
                "seed": "generated_masks",
                "filename": filename,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "intersection": metrics["intersection"],
                "union": metrics["union"],
                "pred_pixels": metrics["pred_pixels"],
                "gt_pixels": metrics["gt_pixels"],
                "mask_path": str(mask_path),
            }
        )

    ious = np.array([float(row["iou"]) for row in per_image_rows])
    dices = np.array([float(row["dice"]) for row in per_image_rows])
    return {
        "seed": "generated_masks",
        "steps": "",
        "sampler_w": "",
        "mpd_w": cli_args.mpd_w,
        "num_generations": "",
        "n": len(per_image_rows),
        "seconds": time.time() - started_at,
        "macro_iou": float(ious.mean()),
        "macro_dice": float(dices.mean()),
        "median_iou": float(np.median(ious)),
        "median_dice": float(np.median(dices)),
        "global_iou": float(total_intersection / total_union),
        "global_dice": float(2 * total_intersection / (total_pred_pixels + total_gt_pixels)),
        "per_image_rows": per_image_rows,
    }


def evaluate_seed(
    cli_args: argparse.Namespace,
    model_args_: argparse.Namespace,
    sampler: torch.nn.Module,
    dataloader: DataLoader,
    dataset: CustomImageDatasetCondition_MPD,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    seed_all(seed)

    per_image_rows: list[dict[str, Any]] = []
    total_intersection = 0
    total_union = 0
    total_pred_pixels = 0
    total_gt_pixels = 0
    started_at = time.time()

    mask_dir = cli_args.output_dir / f"seed_{seed}" / "masks"
    if cli_args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            condition = batch["input_image"].to(device)
            batch_size = condition.size(0)
            channels, height, width = condition.shape[1:]

            initial_noise = torch.randn(
                cli_args.num_generations * batch_size,
                channels,
                height,
                width,
                device=device,
            )
            repeated_condition = condition.repeat(cli_args.num_generations, 1, 1, 1)
            initial_noise = (1 - cli_args.mpd_w) * initial_noise + cli_args.mpd_w * repeated_condition

            x0 = sampler(initial_noise, steps=model_args_.steps)
            mask_prob = torch.sigmoid(x0.mean(dim=1, keepdim=True))

            refined_masks = []
            for idx in range(x0.size(0)):
                soft_mask = mask_prob[idx, 0].detach().cpu().numpy()
                crf_image = repeated_condition[idx].permute(1, 2, 0).detach().cpu().numpy()
                crf_image = ((crf_image + 1) * 127.5).astype(np.uint8)
                refined_masks.append(torch.tensor(apply_crf(crf_image, soft_mask), device=device))

            binary_masks = (torch.stack(refined_masks).unsqueeze(1).float() > 0.5).float()
            voted_masks = torch.zeros(batch_size, 1, height, width, device=device)

            for idx in range(batch_size):
                processed_stack = []
                for generation_idx in range(cli_args.num_generations):
                    mask_idx = generation_idx * batch_size + idx
                    mask_np = binary_masks[mask_idx, 0].cpu().numpy()
                    labeled_array, num_features = ndi.label(mask_np)
                    sizes = ndi.sum(mask_np, labeled_array, range(num_features + 1))
                    if len(sizes) > 1:
                        largest_label = int(np.argmax(sizes[1:]) + 1)
                        largest_component = (labeled_array == largest_label).astype(float)
                    else:
                        largest_component = np.zeros_like(mask_np)
                    filled_component = binary_fill_holes(largest_component).astype(float)
                    processed_stack.append(torch.tensor(filled_component, device=device))

                stacked_tensor = torch.stack(processed_stack)
                voted_masks[idx, 0] = (stacked_tensor.mean(dim=0) > 0.5).float()

            for idx in range(batch_size):
                dataset_idx = batch_idx * cli_args.eval_batch_size + idx
                filename = batch["filename"][idx]
                original_path = dataset.img_path_files[dataset_idx]
                gt_path = dataset.gt_path_files[dataset_idx]
                original_width, original_height = Image.open(original_path).size

                pred = F.interpolate(
                    voted_masks[idx].unsqueeze(0),
                    size=(original_height, original_width),
                    mode="nearest",
                )[0, 0].cpu().numpy() > 0.5
                gt = np.array(Image.open(gt_path).convert("L")) > 127

                metrics = compute_binary_metrics(pred, gt)
                total_intersection += int(metrics["intersection"])
                total_union += int(metrics["union"])
                total_pred_pixels += int(metrics["pred_pixels"])
                total_gt_pixels += int(metrics["gt_pixels"])

                row = {
                    "seed": seed,
                    "filename": filename,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "intersection": metrics["intersection"],
                    "union": metrics["union"],
                    "pred_pixels": metrics["pred_pixels"],
                    "gt_pixels": metrics["gt_pixels"],
                }
                per_image_rows.append(row)

                if cli_args.save_masks:
                    mask_path = mask_dir / f"{filename}_mask_mpd_w{cli_args.mpd_w:g}_seed{seed}.PNG"
                    Image.fromarray((pred.astype(np.uint8) * 255)).save(mask_path)

    ious = np.array([float(row["iou"]) for row in per_image_rows])
    dices = np.array([float(row["dice"]) for row in per_image_rows])
    return {
        "seed": seed,
        "steps": cli_args.steps,
        "sampler_w": cli_args.sampler_w,
        "mpd_w": cli_args.mpd_w,
        "num_generations": cli_args.num_generations,
        "n": len(per_image_rows),
        "seconds": time.time() - started_at,
        "macro_iou": float(ious.mean()),
        "macro_dice": float(dices.mean()),
        "median_iou": float(np.median(ious)),
        "median_dice": float(np.median(dices)),
        "global_iou": float(total_intersection / total_union),
        "global_dice": float(2 * total_intersection / (total_pred_pixels + total_gt_pixels)),
        "per_image_rows": per_image_rows,
    }


def write_seed_csv(output_dir: Path, seed: int, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"per_image_seed_{seed}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_and_write_summary(cli_args: argparse.Namespace, seed_results: list[dict[str, Any]]) -> None:
    summary_rows = []
    for result in seed_results:
        row = {key: value for key, value in result.items() if key != "per_image_rows"}
        summary_rows.append(row)
        print("SEED_RESULT", json.dumps(row, sort_keys=True), flush=True)

    aggregate: dict[str, Any] = {
        "checkpoint": str(cli_args.checkpoint),
        "data_root": str(cli_args.data_root),
        "pred_mask_dir": str(cli_args.pred_mask_dir) if cli_args.pred_mask_dir is not None else "",
        "steps": cli_args.steps,
        "sampler_w": cli_args.sampler_w,
        "mpd_w": cli_args.mpd_w,
        "num_generations": cli_args.num_generations,
        "seeds": cli_args.seeds,
    }
    for metric in ("macro_iou", "macro_dice", "median_iou", "median_dice", "global_iou", "global_dice"):
        values = np.array([float(row[metric]) for row in summary_rows])
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        print(
            "SUMMARY",
            metric,
            "mean",
            aggregate[f"{metric}_mean"],
            "std",
            aggregate[f"{metric}_std"],
            flush=True,
        )

    cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cli_args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({"aggregate": aggregate, "seeds": summary_rows}, indent=2) + "\n")

    csv_path = cli_args.output_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    cli_args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((cli_args.img_size, cli_args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CustomImageDatasetCondition_MPD(cli_args.data_root, "test", transform)
    if len(dataset) == 0:
        raise ValueError(f"No test samples found under {cli_args.data_root}")

    if cli_args.pred_mask_dir is not None:
        print("PRED_MASK_DIR", str(cli_args.pred_mask_dir), flush=True)
        print("DATA_ROOT", str(cli_args.data_root), "samples", len(dataset), flush=True)
        print("PARAMS", "mpd_w", cli_args.mpd_w, flush=True)
        result = evaluate_generated_masks(cli_args, dataset)
        write_seed_csv(cli_args.output_dir, 0, result["per_image_rows"])
        print_and_write_summary(cli_args, [result])
        return

    dataloader = DataLoader(
        dataset,
        batch_size=cli_args.eval_batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
    )
    model_args_, _model, sampler = load_model_and_sampler(cli_args, device)

    print("CHECKPOINT", str(cli_args.checkpoint), flush=True)
    print("DATA_ROOT", str(cli_args.data_root), "samples", len(dataset), flush=True)
    print("PARAMS", "sampler_w", cli_args.sampler_w, "mpd_w", cli_args.mpd_w, flush=True)

    seed_results = []
    for seed in cli_args.seeds:
        result = evaluate_seed(cli_args, model_args_, sampler, dataloader, dataset, device, seed)
        write_seed_csv(cli_args.output_dir, seed, result["per_image_rows"])
        seed_results.append(result)

    print_and_write_summary(cli_args, seed_results)


if __name__ == "__main__":
    main()
