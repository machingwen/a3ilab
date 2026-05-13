#!/usr/bin/env python3
"""Evaluate matched MPD vs gpcd_6ch ISIC checkpoints with DenseCRF.

This script is a standalone version of the direct evaluator used for the
MPD/gpcd_6ch comparison. It intentionally keeps the inference, post-processing,
seeds, and metric definitions unchanged from that evaluator.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
from models.engine import DDIMSamplerImage, DDIMSamplerImageGPCDConcat  # noqa: E402
from utils import get_model  # noqa: E402


DEFAULT_DATA_ROOT = Path("/work-pvc/macw1030/isic_mpd_png")
DEFAULT_MPD_CKPT = (
    REPO_ROOT
    / "checkpoint/isic_mpd_matched_128_e500/w_0.9_lr5.0e-06/"
    / "model_epoch500_lr5.0e-06_w0.9.pth"
)
DEFAULT_GPCD6_CKPT = (
    REPO_ROOT
    / "checkpoint/isic_mpd_matched_128_e500/gpcd_6ch_lr5.0e-06/"
    / "model_epoch500_lr5.0e-06_gpcd_6ch.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MPD and gpcd_6ch ISIC checkpoints with DenseCRF."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mpd_checkpoint", type=Path, default=DEFAULT_MPD_CKPT)
    parser.add_argument("--gpcd6_checkpoint", type=Path, default=DEFAULT_GPCD6_CKPT)
    parser.add_argument("--methods", nargs="+", default=["mpd", "gpcd_6ch"], choices=["mpd", "gpcd_6ch"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mpd_w", type=float, default=0.9)
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


def model_args(method: str, steps: int, img_size: int, eval_batch_size: int, num_workers: int) -> argparse.Namespace:
    return argparse.Namespace(
        arch="unetattention_image",
        method=method,
        img_size=img_size,
        num_timestep=1000,
        beta=(0.0001, 0.02),
        num_condition=[57, 1],
        emb_size=128,
        channel_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        use_spatial_transformer=False,
        num_heads=4,
        num_sample=4,
        w=3,
        projection_dim=512,
        only_table=False,
        concat=False,
        only_encoder=False,
        num_head_channels=-1,
        encoder_path=None,
        steps=steps,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        ignored=None,
    )


def load_model_and_sampler(
    method: str,
    checkpoint_path: Path,
    device: torch.device,
    steps: int,
    img_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[argparse.Namespace, torch.nn.Module, torch.nn.Module]:
    args = model_args(method, steps, img_size, eval_batch_size, num_workers)
    checkpoint = torch.load(checkpoint_path, map_location=device)["model"]
    state_dict = OrderedDict(
        (key[7:] if key.startswith("module") else key, value)
        for key, value in checkpoint.items()
    )

    model = get_model(args)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    if method == "mpd":
        sampler = DDIMSamplerImage(model=model, beta=args.beta, T=args.num_timestep, w=args.w).to(device)
    else:
        sampler = DDIMSamplerImageGPCDConcat(model=model, beta=args.beta, T=args.num_timestep, w=args.w).to(device)

    return args, model, sampler


def evaluate_seed(
    method: str,
    sampler: torch.nn.Module,
    args: argparse.Namespace,
    dataloader: DataLoader,
    dataset: CustomImageDatasetCondition_MPD,
    device: torch.device,
    seed: int,
    mpd_w: float,
) -> dict[str, Any]:
    seed_all(seed)

    ious: list[float] = []
    dices: list[float] = []
    total_intersection = 0
    total_union = 0
    total_pred_pixels = 0
    total_gt_pixels = 0
    best: tuple[str, float, float] | None = None
    worst: tuple[str, float, float] | None = None
    num_generations = 3
    started_at = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            condition = batch["input_image"].to(device)
            batch_size = condition.size(0)
            channels, height, width = condition.shape[1:]

            initial_noise = torch.randn(
                num_generations * batch_size,
                channels,
                height,
                width,
                device=device,
            )
            repeated_condition = condition.repeat(num_generations, 1, 1, 1)

            if method == "mpd":
                initial_noise = (1 - mpd_w) * initial_noise + mpd_w * repeated_condition

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if method == "mpd":
                    x0 = sampler(initial_noise, steps=args.steps)
                else:
                    x0 = sampler(initial_noise, repeated_condition, steps=args.steps)

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
                for gen_idx in range(num_generations):
                    mask_idx = gen_idx * batch_size + idx
                    mask_np = binary_masks[mask_idx, 0].cpu().numpy()
                    labeled_array, num_features = ndi.label(mask_np)
                    sizes = ndi.sum(mask_np, labeled_array, range(num_features + 1))
                    if len(sizes) > 1:
                        largest_label = np.argmax(sizes[1:]) + 1
                        largest_component = (labeled_array == largest_label).astype(float)
                    else:
                        largest_component = np.zeros_like(mask_np)
                    filled_component = binary_fill_holes(largest_component).astype(float)
                    processed_stack.append(torch.tensor(filled_component, device=device))

                stacked_tensor = torch.stack(processed_stack)
                voted_masks[idx, 0] = (stacked_tensor.mean(dim=0) > 0.5).float()

            for idx in range(batch_size):
                filename = batch["filename"][idx]
                gt_path = Path(dataset.gt_path) / f"{filename}.PNG"
                original_path = dataset.img_path_files[batch_idx * args.eval_batch_size + idx]
                original_width, original_height = Image.open(original_path).size

                pred = F.interpolate(
                    voted_masks[idx].unsqueeze(0),
                    size=(original_height, original_width),
                    mode="nearest",
                )[0, 0].cpu().numpy() > 0.5
                gt = np.array(Image.open(gt_path).convert("L")) > 127

                intersection = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                pred_pixels = pred.sum()
                gt_pixels = gt.sum()

                iou = intersection / union if union else 1.0
                dice = 2 * intersection / (pred_pixels + gt_pixels) if pred_pixels + gt_pixels else 1.0

                ious.append(float(iou))
                dices.append(float(dice))
                total_intersection += int(intersection)
                total_union += int(union)
                total_pred_pixels += int(pred_pixels)
                total_gt_pixels += int(gt_pixels)

                record = (filename, float(iou), float(dice))
                best = record if best is None or iou > best[1] else best
                worst = record if worst is None or iou < worst[1] else worst

    # Macro IoU/Dice: arithmetic mean of per-image IoU/Dice over the test set.
    macro_iou = float(np.mean(ious))
    macro_dice = float(np.mean(dices))
    # Median IoU/Dice: median of per-image IoU/Dice over the test set.
    median_iou = float(np.median(ious))
    median_dice = float(np.median(dices))
    # Global IoU/Dice: compute one aggregate score from summed intersections/unions/pixel counts.
    global_iou = float(total_intersection / total_union)
    global_dice = float(2 * total_intersection / (total_pred_pixels + total_gt_pixels))

    return {
        "seed": seed,
        "steps": args.steps,
        "n": len(ious),
        "seconds": time.time() - started_at,
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "median_iou": median_iou,
        "median_dice": median_dice,
        "global_iou": global_iou,
        "global_dice": global_dice,
        "best": best,
        "worst": worst,
    }


def print_summary(method: str, rows: list[dict[str, Any]]) -> None:
    for metric in [
        "macro_iou",
        "macro_dice",
        "median_iou",
        "median_dice",
        "global_iou",
        "global_dice",
    ]:
        values = np.array([row[metric] for row in rows])
        print(
            "SUMMARY",
            method,
            metric,
            "mean",
            float(values.mean()),
            "std",
            float(values.std(ddof=1)),
            flush=True,
        )


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
    dataloader = DataLoader(
        dataset,
        batch_size=cli_args.eval_batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
    )

    checkpoint_by_method = {
        "mpd": cli_args.mpd_checkpoint,
        "gpcd_6ch": cli_args.gpcd6_checkpoint,
    }

    for method in cli_args.methods:
        checkpoint = checkpoint_by_method[method]
        args, _model, sampler = load_model_and_sampler(
            method,
            checkpoint,
            device,
            cli_args.steps,
            cli_args.img_size,
            cli_args.eval_batch_size,
            cli_args.num_workers,
        )
        rows = []
        print("METHOD", method, "checkpoint", str(checkpoint), flush=True)
        for seed in cli_args.seeds:
            row = evaluate_seed(method, sampler, args, dataloader, dataset, device, seed, cli_args.mpd_w)
            rows.append(row)
            print("SEED_RESULT", row, flush=True)
        print_summary(method, rows)
        print("", flush=True)


if __name__ == "__main__":
    main()
