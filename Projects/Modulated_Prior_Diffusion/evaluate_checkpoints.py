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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_test import CustomImageDatasetCondition_MPD
from models.engine import DDIMSamplerImage
from utils import get_model


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
    projection_dim = 512
    only_table = False
    concat = False
    num_head_channels = -1
    ignored = None


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def largest_component(mask):
    labeled_array, num_features = ndi.label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labeled_array, range(num_features + 1))
    largest_label = int(np.argmax(sizes[1:]) + 1)
    return labeled_array == largest_label


def postprocess_mask(mask):
    mask = largest_component(mask)
    return binary_fill_holes(mask).astype(bool)


def binarize_gt(gt_tensor):
    gt = gt_tensor.mean(dim=0).detach().cpu().numpy()
    gt = (gt + 1.0) * 0.5
    return gt > 0.5


def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return dice, iou, precision, recall


def load_model(checkpoint_path, device, model_args):
    model = get_model(model_args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(strip_module_prefix(checkpoint["model"]))
    model.eval().to(device)
    return model


def evaluate_checkpoint(checkpoint_path, dataloader, device, args, model_args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(checkpoint_path, device, model_args)
    sampler = DDIMSamplerImage(
        model=model,
        beta=model_args.beta,
        T=model_args.num_timestep,
        w=args.guidance_w,
    ).to(device)

    rows = []
    totals = {"dice": [], "iou": [], "precision": [], "recall": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=os.path.basename(checkpoint_path)):
            c2_image = batch["input_image"].to(device, non_blocking=True)
            gt_image = batch["groundtruth_image"]
            filenames = batch["filename"]

            batch_size, channels, height, width = c2_image.shape
            noise = torch.randn(
                args.num_generations * batch_size,
                channels,
                height,
                width,
                device=device,
            )
            c2_repeat = c2_image.repeat(args.num_generations, 1, 1, 1)
            x_i = (1 - args.mpd_w) * noise + args.mpd_w * c2_repeat
            x0_all = sampler(x_i, steps=args.steps)

            probs = ((x0_all.mean(dim=1, keepdim=True) + 1.0) * 0.5).clamp(0.0, 1.0)
            preds = probs > args.threshold

            for idx in range(batch_size):
                pred_stack = []
                for generation_idx in range(args.num_generations):
                    sample_idx = generation_idx * batch_size + idx
                    pred_np = preds[sample_idx, 0].detach().cpu().numpy()
                    if args.postprocess:
                        pred_np = postprocess_mask(pred_np)
                    pred_stack.append(torch.tensor(pred_np, device=device))

                pred_mean = torch.stack(pred_stack).float().mean(dim=0)
                pred_final = (pred_mean > 0.5).detach().cpu().numpy()
                gt_final = binarize_gt(gt_image[idx])
                dice, iou, precision, recall = compute_metrics(pred_final, gt_final)

                row = {
                    "filename": filenames[idx],
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                }
                rows.append(row)
                totals["dice"].append(dice)
                totals["iou"].append(iou)
                totals["precision"].append(precision)
                totals["recall"].append(recall)

    summary = {name: float(np.mean(values)) for name, values in totals.items()}
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/work/macw1030/isic_mpd_png")
    parser.add_argument("--checkpoint_root", default="checkpoint/MED_pth_w_ISIC/w_0.9_lr5.0e-06")
    parser.add_argument("--epochs", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--mpd_w", type=float, default=0.9)
    parser.add_argument("--guidance_w", type=float, default=5.0)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--output_csv", default="result/MED_pth_w_ISIC/evaluation_metrics.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = ModelArgs()
    model_args.img_size = args.img_size

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CustomImageDatasetCondition_MPD(args.data_root, "test", transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    all_rows = []
    summaries = []

    for epoch in args.epochs:
        checkpoint = os.path.join(
            args.checkpoint_root,
            f"model_epoch{epoch}_lr{args.lr:.1e}_w{args.mpd_w:.1f}.pth",
        )
        if not os.path.exists(checkpoint):
            print(f"Skipping missing checkpoint: {checkpoint}")
            continue
        summary, rows = evaluate_checkpoint(checkpoint, dataloader, device, args, model_args)
        summary["epoch"] = epoch
        summaries.append(summary)
        for row in rows:
            row["epoch"] = epoch
            all_rows.append(row)
        print(
            f"epoch {epoch}: "
            f"dice={summary['dice']:.4f}, "
            f"iou={summary['iou']:.4f}, "
            f"precision={summary['precision']:.4f}, "
            f"recall={summary['recall']:.4f}"
        )

    with open(args.output_csv, "w", newline="") as csv_file:
        fieldnames = ["epoch", "filename", "dice", "iou", "precision", "recall"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    if summaries:
        best = max(summaries, key=lambda item: item["dice"])
        print(
            f"best_by_dice epoch {best['epoch']}: "
            f"dice={best['dice']:.4f}, iou={best['iou']:.4f}"
        )
    print(f"saved per-image metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
