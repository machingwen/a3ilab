import argparse
import os
import time
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_test import CustomImageDatasetCondition_MPD
from models.engine import DDIMSamplerImage, DDIMSamplerImageGPCDConcat
from train_main import append_metrics_csv, mask_metrics, maybe_limit_dataset
from utils import get_model


class ForwardProfiler:
    def __init__(self, module, device):
        self.module = module
        self.device = device
        self.calls = 0
        self.item_calls = 0
        self.seconds = 0.0
        self._original_forward = module.forward

    def __enter__(self):
        def profiled_forward(*args, **kwargs):
            batch_size = 0
            if args and torch.is_tensor(args[0]) and args[0].ndim > 0:
                batch_size = args[0].shape[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            out = self._original_forward(*args, **kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.seconds += time.perf_counter() - start_time
            self.calls += 1
            self.item_calls += batch_size
            return out

        self.module.forward = profiled_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        self.module.forward = self._original_forward
        return False


def default_checkpoint_path(args, method):
    lr = f"{args.lr:.1e}"
    if method == "mpd":
        method_dir = f"w_{args.w:.1f}_lr{lr}"
        ckpt_name = f"model_epoch{args.epoch}_lr{lr}_w{args.w:.1f}.pth"
    else:
        method_dir = f"{method}_lr{lr}"
        ckpt_name = f"model_epoch{args.epoch}_lr{lr}_{method}.pth"
    return os.path.join(args.checkpoint_root, method_dir, ckpt_name)


def method_result_dir(args, method):
    lr = f"{args.lr:.1e}"
    if method == "mpd":
        return os.path.join("result", args.exp, f"w_{args.w:.1f}_lr{lr}")
    return os.path.join("result", args.exp, f"{method}_lr{lr}")


def load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)["model"]
    state = OrderedDict()
    for key, value in ckpt.items():
        state[key[7:] if key.startswith("module.") else key] = value
    return state


def build_sampler(args, method, model, device, steps):
    if args.sample_method != "ddim":
        raise ValueError("eval_isic_metrics.py supports --sample_method ddim for MPD/GPCD comparisons")
    if method == "gpcd_concat":
        return DDIMSamplerImageGPCDConcat(
            model=model,
            beta=args.beta,
            T=args.num_timestep,
            w=args.w,
        ).to(device)
    return DDIMSamplerImage(
        model=model,
        beta=args.beta,
        T=args.num_timestep,
        w=args.w,
    ).to(device)


def evaluate_method(args, method, device, dataloader, sample_shape):
    checkpoint_path = args.mpd_checkpoint_path if method == "mpd" else args.gpcd_checkpoint_path
    checkpoint_path = checkpoint_path or default_checkpoint_path(args, method)

    args.method = method
    model = get_model(args)
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    result_dir = method_result_dir(args, method)
    metrics_path = os.path.join(result_dir, "metrics.csv")

    for steps in args.step_values:
        torch.manual_seed(args.seed + steps)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + steps)
        sampler = build_sampler(args, method, model, device, steps)
        sampler.eval()
        iou_sum = 0.0
        dice_sum = 0.0
        eval_count = 0
        inference_seconds = 0.0
        sampler_calls = 0
        denoiser = model.denoiser if hasattr(model, "denoiser") else model

        with torch.no_grad(), ForwardProfiler(denoiser, device) as profiler:
            for batch in dataloader:
                c2_image = batch["input_image"].to(device)
                target_image = batch["groundtruth_image"].to(device)
                batch_size = c2_image.shape[0]
                x_i = torch.randn(batch_size, *sample_shape, device=device)
                if method == "mpd":
                    x_i = (1 - args.w) * x_i + args.w * c2_image

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                if method == "gpcd_concat":
                    pred = sampler(x_i, c2_image, steps=steps)
                else:
                    pred = sampler(x_i, steps=steps)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - start_time
                sampler_calls += 1

                pred = torch.clamp(pred * 0.5 + 0.5, 0.0, 1.0)
                target = torch.clamp(target_image * 0.5 + 0.5, 0.0, 1.0)
                iou, dice = mask_metrics(pred, target)
                iou_sum += iou.sum().item()
                dice_sum += dice.sum().item()
                eval_count += batch_size

        mean_iou = iou_sum / max(1, eval_count)
        mean_dice = dice_sum / max(1, eval_count)
        inference_time_per_image = inference_seconds / max(1, eval_count)
        denoiser_calls_per_image = profiler.item_calls / max(1, eval_count)
        avg_denoiser_forward = profiler.seconds / max(1, profiler.calls)
        append_metrics_csv(metrics_path, {
            "phase": "steps_eval",
            "epoch": args.epoch,
            "method": method,
            "mpd_w": f"{args.w:.6g}" if method == "mpd" else "",
            "steps": steps,
            "train_loss": "",
            "iou": f"{mean_iou:.6g}",
            "dice": f"{mean_dice:.6g}",
            "inference_time_per_image_sec": f"{inference_time_per_image:.6g}",
            "eval_samples": eval_count,
            "checkpoint_path": checkpoint_path,
            "total_inference_time_sec": f"{inference_seconds:.6g}",
            "sampler_calls": sampler_calls,
            "denoiser_forward_calls": profiler.calls,
            "denoiser_calls_per_image": f"{denoiser_calls_per_image:.6g}",
            "avg_time_per_denoiser_forward_sec": f"{avg_denoiser_forward:.6g}",
            "sampler": args.sample_method,
        })
        print(
            f"steps_eval method={method} epoch={args.epoch} steps={steps} "
            f"eval_samples={eval_count} iou={mean_iou:.6g} dice={mean_dice:.6g} "
            f"inference_time_per_image_sec={inference_time_per_image:.6g} "
            f"sampler_calls={sampler_calls} denoiser_forward_calls={profiler.calls} "
            f"avg_time_per_denoiser_forward_sec={avg_denoiser_forward:.6g} "
            f"metrics_csv={metrics_path}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="isic_core")
    parser.add_argument("--eval_data_root", type=str, required=True)
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument("--mpd_checkpoint_path", type=str, default=None)
    parser.add_argument("--gpcd_checkpoint_path", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w", type=float, default=0.9)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--step_values", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--sample_method", type=str, default="ddim", choices=["ddim"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--methods", type=str, nargs="+", default=["mpd", "gpcd_concat"], choices=["mpd", "gpcd_concat"])
    parser.add_argument("--arch", type=str, default="unetattention_image")
    parser.add_argument("--num_timestep", type=int, default=1000)
    parser.add_argument("--beta", type=Tuple[float, float], default=(0.0001, 0.02))
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--num_condition", type=int, nargs="+", default=None)
    parser.add_argument("--use_spatial_transformer", action="store_true")
    parser.add_argument("--only_table", action="store_true")
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--only_encoder", action="store_true")
    parser.add_argument("--encoder_path", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join("checkpoint", args.exp)
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_ds = CustomImageDatasetCondition_MPD(
        data_root=args.eval_data_root,
        dataset_type="test",
        transform=transform,
    )
    eval_ds = maybe_limit_dataset(eval_ds, args.max_eval_samples)
    if len(eval_ds) == 0:
        raise ValueError(f"No evaluation samples found under {args.eval_data_root}")
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sample_shape = (3, args.img_size, args.img_size)
    print(
        f"eval_only device={device} eval_samples={len(eval_ds)} "
        f"eval_batch_size={args.eval_batch_size} steps={args.step_values}"
    )
    for method in args.methods:
        evaluate_method(args, method, device, dataloader, sample_shape)


if __name__ == "__main__":
    main()
