# generate_main

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
import shutil
from typing import Tuple
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from scipy import ndimage as ndi

import torch
import numpy as np
import scipy.ndimage as ndi




from models.embedding import *
from models.engine import ConditionalGaussianDiffusionTrainer, ConditionalDiffusionEncoderTrainer, DDIMSampler, DDIMSamplerEncoder,ImageConditionalGaussianDiffusionTrainer,DDIMSamplerImage,DDIMSamplerImageGPCDConcat 
#from dataset import CustomImageDataset, CustomSampler
from dataset_test import CustomImageDatasetCondition_MPD, CustomSampler
from utils import GradualWarmupScheduler, get_model, get_optimizer, get_piecewise_constant_schedule, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, LoadEncoder
from scipy.ndimage import binary_fill_holes

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    dcrf = None
    unary_from_softmax = None


def apply_crf(img, prob_map):
    """
    使用 DenseCRF 進行分割後處理
    img: 原始輸入影像 (H, W, 3)
    prob_map: soft mask (H, W), 值域 [0,1]
    回傳 refined mask (H, W)，值為 {0,1}
    """
    if dcrf is None or unary_from_softmax is None:
        raise ImportError("pydensecrf is required for CRF post-processing. Install it or use --smoke_test.")
    # 這兩行是關鍵
    img = np.ascontiguousarray(img, dtype=np.uint8)       # 確保原圖 C-contiguous
    #soft_mask = np.ascontiguousarray(prob_map, dtype=np.float32)  # 確保 soft mask C-contiguous
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)
    
    H, W = prob_map.shape
    d = dcrf.DenseCRF2D(W, H, 2)  # 2分類：前景+背景
    
    # === 增強 soft mask 對比度，避免邊界被吃掉 ===
    prob_map = np.clip(prob_map, 0.001, 0.999)
    # Guard constant masks so CRF normalization does not produce NaNs.
    denom = prob_map.max() - prob_map.min()
    prob_map = (prob_map - prob_map.min()) / (denom if denom > 0 else 1.0)

    # 準備 softmax probability
    probs = np.zeros((2, H, W), dtype=np.float32)
    probs[1, :, :] = prob_map       # 前景
    probs[0, :, :] = 1 - prob_map   # 背景
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

#     # 加入位置平滑
#     d.addPairwiseGaussian(sxy=2, compat=1)
#     # 加入顏色邊界
#     d.addPairwiseBilateral(sxy=20, srgb=5, rgbim=img, compat=5)

#     Q = d.inference(3)  # 迭代次數
    # === DenseCRF 設定 ===
    # 空間平滑適中，讓小雜訊被抑制，但不會過度吃掉邊界
    d.addPairwiseGaussian(sxy=2, compat=1)

    # 顏色邊界權重較高，讓病灶邊緣更貼近實際
    d.addPairwiseBilateral(sxy=20, srgb=5, rgbim=img, compat=5)

    # 增加迭代次數，讓結果更穩定
    Q = d.inference(3)

    refined = np.argmax(Q, axis=0).reshape((H, W))
    return refined


def weight_label(value):
    return f"{float(value):g}"



class Args(argparse.Namespace):
    arch = "unetattention_image"
    method = "mpd"
    img_size = 128
    num_timestep = 1000
    beta = (0.0001, 0.02)
    num_condition = [57, 1]
    emb_size = 128
    channel_mult = [1, 2, 2, 2]
    num_res_blocks = 2
    use_spatial_transformer = False
    num_heads = 4
    num_sample = 4  # 一次生成數量
    sampler_w = 3.0
    w = sampler_w
    mpd_w_values = [0.9]
    projection_dim = 512
    only_table = False
    concat = False
    only_encoder = False
    num_head_channels = -1
    encoder_path = None
    steps = 10
    eval_batch_size = 8
    num_workers = 4
    ignored = None
    data_root = "/work-pvc/macw1030/isic_mpd_png_128"
    checkpoint_root = os.path.join(REPO_ROOT, "checkpoint", "isic_original_mpd_repro_128_e500")
    output_dir = os.path.join(REPO_ROOT, "generate_result", "repro_mpd_postprocess_densecrf")
    checkpoint_path = None
    epoch = 500
    ckpt_lr = "5.0e-06"
    smoke_test = False
    smoke_samples = 2
    no_wandb = False

# 初始化裝置和參數
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = Args()
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default=args.method, choices=["mpd", "gpcd_concat"], help="conditioning method")
parser.add_argument("--data_root", type=str, default=args.data_root, help="test dataset root")
parser.add_argument("--checkpoint_root", type=str, default=args.checkpoint_root, help="root directory containing method checkpoints")
parser.add_argument("--checkpoint_path", type=str, default=args.checkpoint_path, help="explicit checkpoint path")
parser.add_argument("--output_dir", type=str, default=args.output_dir, help="directory for generated outputs")
parser.add_argument("--epoch", type=int, default=args.epoch, help="checkpoint epoch")
parser.add_argument("--ckpt_lr", type=str, default=args.ckpt_lr, help="learning-rate string used in checkpoint filenames")
parser.add_argument("--img_size", type=int, default=args.img_size, help="inference image size")
parser.add_argument("--num_timestep", type=int, default=args.num_timestep, help="number of diffusion timesteps")
parser.add_argument("--steps", type=int, default=args.steps, help="DDIM sampling steps")
parser.add_argument("--eval_batch_size", type=int, default=args.eval_batch_size, help="evaluation batch size")
parser.add_argument("--num_workers", type=int, default=args.num_workers, help="number of dataloader workers")
parser.add_argument("--sampler_w", type=float, default=args.sampler_w, help="DDIM sampler guidance weight; distinct from MPD conditioning weight")
parser.add_argument("--mpd_w_values", type=float, nargs="+", default=args.mpd_w_values, help="MPD conditioning mix weights")
parser.add_argument("--emb_size", type=int, default=args.emb_size, help="embedding output dimension")
parser.add_argument("--channel_mult", type=int, nargs="+", default=args.channel_mult, help="width of unet model")
parser.add_argument("--num_res_blocks", type=int, default=args.num_res_blocks, help="number of residual blocks")
parser.add_argument("--projection_dim", type=int, default=args.projection_dim, help="attention projection dimension")
parser.add_argument("--num_heads", type=int, default=args.num_heads, help="number of attention heads")
parser.add_argument("--num_head_channels", type=int, default=args.num_head_channels, help="attention head channels")
parser.add_argument("--smoke_test", action="store_true", help="run a tiny synthetic inference smoke test without a checkpoint")
parser.add_argument("--smoke_samples", type=int, default=args.smoke_samples, choices=[1, 2, 3, 4], help="number of synthetic smoke-test samples")
parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
cli_args, _ = parser.parse_known_args()
args.method = cli_args.method
args.data_root = cli_args.data_root
args.checkpoint_root = cli_args.checkpoint_root
args.checkpoint_path = cli_args.checkpoint_path
args.output_dir = cli_args.output_dir
args.epoch = cli_args.epoch
args.ckpt_lr = cli_args.ckpt_lr
args.img_size = cli_args.img_size
args.num_timestep = cli_args.num_timestep
args.steps = cli_args.steps
args.eval_batch_size = cli_args.eval_batch_size
args.num_workers = cli_args.num_workers
args.sampler_w = cli_args.sampler_w
args.w = cli_args.sampler_w
args.mpd_w_values = cli_args.mpd_w_values
args.emb_size = cli_args.emb_size
args.channel_mult = cli_args.channel_mult
args.num_res_blocks = cli_args.num_res_blocks
args.projection_dim = cli_args.projection_dim
args.num_heads = cli_args.num_heads
args.num_head_channels = cli_args.num_head_channels
args.smoke_test = cli_args.smoke_test
args.smoke_samples = cli_args.smoke_samples
args.no_wandb = cli_args.no_wandb
mpd_w_values = args.mpd_w_values if args.method == "mpd" else [None]

for mpd_w in mpd_w_values:
# 加載模型
    if args.method == "mpd":
        mpd_w = round(mpd_w, 1)
        mpd_w_str = weight_label(mpd_w)
    model = get_model(args)
    epoch = args.epoch
    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    elif args.method == "mpd":
        ckpt_path = os.path.join(
            args.checkpoint_root,
            f"mpd_w_{mpd_w_str}_lr{args.ckpt_lr}",
            f"model_epoch{epoch}_lr{args.ckpt_lr}_mpd_w{mpd_w_str}.pth",
        )
        legacy_ckpt_path = os.path.join(
            args.checkpoint_root,
            f"w_{mpd_w_str}_lr{args.ckpt_lr}",
            f"model_epoch{epoch}_lr{args.ckpt_lr}_w{mpd_w_str}.pth",
        )
        if not os.path.exists(ckpt_path) and os.path.exists(legacy_ckpt_path):
            ckpt_path = legacy_ckpt_path
    else:
        ckpt_path = os.path.join(args.checkpoint_root, f"{args.method}_lr{args.ckpt_lr}", f"model_epoch{epoch}_lr{args.ckpt_lr}_{args.method}.pth")
    if args.smoke_test:
        print("smoke_test=True; skipping checkpoint load.")
        ckpt = None
    else:
        ckpt = torch.load(ckpt_path, map_location=device)["model"]
    



    if ckpt is not None:
        new_dict = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        try:
            model.load_state_dict(new_dict)
            print("All keys successfully match")
        except:
            print("some keys are missing!")

    for p in model.parameters():
        p.requires_grad = False



    model.eval()
    model.to(device)
    denoiser = model.denoiser if hasattr(model, "denoiser") else model
    denoiser_params = sum(p.numel() for p in denoiser.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"method={args.method} denoiser_params={denoiser_params:,} total_params={total_params:,}")

    sampler = DDIMSamplerImage(
        model=model,
        beta=args.beta,
        T=args.num_timestep,
        w=args.w,
    ).to(device)
    if args.method == "gpcd_concat":
        sampler = DDIMSamplerImageGPCDConcat(
            model=model,
            beta=args.beta,
            T=args.num_timestep,
            w=args.w,
        ).to(device)

    if args.encoder_path is not None and args.method == "mpd":
        encoder = LoadEncoder(args).to(device)
        sampler = DDIMSamplerEncoder(
            model=model,
            encoder=encoder,
            beta=args.beta,
            T=args.num_timestep,
            w=args.w,
            only_encoder=args.only_encoder
        ).to(device)




    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.smoke_test:
        B = args.smoke_samples
        C = 3
        H = args.img_size
        W = args.img_size
        c2_image = torch.randn(B, C, H, W, device=device)
        x_i = torch.randn(B, C, H, W, device=device)
        if args.method == "mpd":
            x_i = (1 - mpd_w) * x_i + mpd_w * c2_image
            x0_all = sampler(x_i, steps=args.steps)
        else:
            x0_all = sampler(x_i, c2_image, steps=args.steps)
        print(f"smoke_test output shape: {tuple(x0_all.shape)}")
        raise SystemExit(0)

    # 構建數據加載器
    valid_ds = CustomImageDatasetCondition_MPD(data_root=args.data_root, dataset_type="test", transform=transform)
    if len(valid_ds) == 0:
        raise ValueError(f"No evaluation samples found under {args.data_root}")
    dataloader_val = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    # 從數據加載器中獲取一個批次的數據
    # Disabled: the generation loop below initializes the DataLoader itself.
    # Fetching a throwaway batch here starts extra workers and can add shared-memory pressure.
    # _, x_sample = next(enumerate(dataloader_val))

    # 創建圖片儲存目錄
    save_dir = os.path.join(args.output_dir, f"mpd_w_{mpd_w_str}" if args.method == "mpd" else args.method)
    os.makedirs(save_dir, exist_ok=True)

    shutil.rmtree(os.path.join(save_dir, ".ipynb_checkpoints"), ignore_errors=True)

    results = []
    num_generations = 3  # 可改為 5

    for batch_idx, x_sample in enumerate(dataloader_val):
        c2_image = x_sample["input_image"].to(device)
        c3_image = x_sample["groundtruth_image"].to(device)

        B = c2_image.size(0)
        C, H, W = c2_image.shape[1:]

        # 一次生成 num_generations * B 張
        x_i = torch.randn(num_generations * B, C, H, W).to(device)
        c2_image_repeat = c2_image.repeat(num_generations, 1, 1, 1)
        if args.method == "mpd":
            x_i = (1 - mpd_w) * x_i + mpd_w * c2_image_repeat

        # Sampling
        if args.method == "mpd":
            x0_all = sampler(x_i, steps=args.steps)
        else:
            x0_all = sampler(x_i, c2_image_repeat, steps=args.steps)
        x0_mask_logit = x0_all.mean(dim=1, keepdim=True)   # (N,1,H,W)
        x0_mask_prob  = torch.sigmoid(x0_mask_logit)       # (N,1,H,W)

        
        
        # === 加入 CRF refine (在 soft mask 上) ===
        x0_all_refined = []
        for idx in range(x0_all.size(0)):
            # 取得 soft mask 並 resize 成原圖大小
            #soft_mask = x0_all[idx, 0].detach().cpu().numpy()#有問題改回這個
            #soft_mask = torch.sigmoid(x0_all[idx, 0]).detach().cpu().numpy()
            soft_mask = x0_mask_prob[idx, 0].detach().cpu().numpy()
            orig_img = c2_image_repeat[idx].permute(1,2,0).detach().cpu().numpy()
            orig_img = ((orig_img + 1) * 127.5).astype(np.uint8)  # 還原到 [0,255]
            refined_mask = apply_crf(orig_img, soft_mask)
            x0_all_refined.append(torch.tensor(refined_mask, device=device))

        # 把 CRF 輸出轉回 tensor
        x0_all_refined = torch.stack(x0_all_refined).unsqueeze(1).float()
        
        
        
        x0_all_bin = (x0_all_refined > 0.5).float()

        # 初始化聯集張量
        x0_union = torch.zeros(B, 1, H, W, device=device)

        for idx in range(B):
            processed_stack = []
            for i in range(num_generations):
                img_idx = i * B + idx
                x0_np = x0_all_bin[img_idx, 0].cpu().numpy()

                # Connected components - keep only the largest white area
                labeled_array, num_features = ndi.label(x0_np)
                sizes = ndi.sum(x0_np, labeled_array, range(num_features + 1))
                if len(sizes) > 1:
                    largest_label = np.argmax(sizes[1:]) + 1  # skip background label 0
                    large_component = (labeled_array == largest_label).astype(float)
                else:
                    large_component = np.zeros_like(x0_np)  # in case of no foreground

                # Fill all holes inside the largest component
                filled_component = binary_fill_holes(large_component).astype(float)

                processed_stack.append(torch.tensor(filled_component, device=device))

            # Union of 3 images
#             stacked_tensor = torch.stack(processed_stack)
#             x0_union[idx, 0] = torch.max(stacked_tensor, dim=0)[0]
            # Average of 3 images + threshold (majority vote)
            stacked_tensor = torch.stack(processed_stack)      # (3, H, W)
            x0_mean = torch.mean(stacked_tensor, dim=0)        # (H, W)
            x0_union[idx, 0] = (x0_mean > 0.5).float()

        print("x0_union.size(0)", x0_union.size(0))

        for idx in range(x0_union.size(0)):
            filename = x_sample["filename"][idx]
            dataset_idx = batch_idx * args.eval_batch_size + idx
            original_image_path = valid_ds.img_path_files[dataset_idx]
            original_image = Image.open(original_image_path).convert("RGB")
            original_size = original_image.size
            original_size_hw = (original_size[1], original_size[0])

            x0_union_resized = F.interpolate(
                x0_union[idx].unsqueeze(0),
                size=original_size_hw,
                mode="nearest",
                #align_corners=False
            )
            x0_union_resized_binary = (x0_union_resized > 0.5).float()
            x0_union_resized_np = (x0_union_resized_binary[0, 0].cpu().numpy() * 255).astype(np.uint8)
            print(f"Original size: {original_size_hw}, Resized mask size: {x0_union_resized_np.shape}")

            
            gt_mask_path = valid_ds.gt_path_files[dataset_idx]
            gt_image = Image.open(gt_mask_path).convert("L")
            gt_image = np.array(gt_image)

            original_image_np = np.array(original_image)
            skin_applied_result = np.zeros_like(original_image_np)
            skin_applied_result[x0_union_resized_np == 255] = original_image_np[x0_union_resized_np == 255]

            num_white_pixels = np.sum(x0_union_resized_np == 255)
            skin_pixels = original_image_np[x0_union_resized_np == 255]

            if num_white_pixels > 0:
                rgb_averages = np.mean(skin_pixels, axis=1)
                Region0, Region1, Region2, Region3, Region4 = 0, 0, 0, 0, 0
                for avg in rgb_averages:
                    if 0 <= avg < 61:
                        Region4 += 1
                    elif 61 <= avg < 121:
                        Region3 += 1
                    elif 121 <= avg < 181:
                        Region2 += 1
                    elif 181 <= avg < 236:
                        Region1 += 1
                    elif 236 <= avg <= 255:
                        Region0 += 1
                TotalPixel = Region0 + Region1 + Region2 + Region3 + Region4
                PixelUnderConsideration = TotalPixel - Region0

                PercentRegion4 = (Region4 / PixelUnderConsideration) * 100 if PixelUnderConsideration > 0 else 0
                PercentRegion3 = (Region3 / PixelUnderConsideration) * 100 if PixelUnderConsideration > 0 else 0
                PercentRegion2 = (Region2 / PixelUnderConsideration) * 100 if PixelUnderConsideration > 0 else 0
                PercentRegion1 = (Region1 / PixelUnderConsideration) * 100 if PixelUnderConsideration > 0 else 0

                Score = (PercentRegion4 / 100) * 4 + (PercentRegion3 / 100) * 3 + (PercentRegion2 / 100) * 2 + (PercentRegion1 / 100) * 1
                print(f"Pigmentation score for {filename}: {Score}")

                results.append({
                    "Image Name": filename,
                    "Pigmentation Score": Score
                })

            # 儲存 Mask 圖與可視化圖
            masks_dir = os.path.join(save_dir, f"masks_mpd_w_{mpd_w_str}" if args.method == "mpd" else f"masks_{args.method}")
            os.makedirs(masks_dir, exist_ok=True)
            mask_suffix = f"mpd_w{mpd_w_str}" if args.method == "mpd" else args.method
            mask_save_path = os.path.join(masks_dir, f"{filename}_mask_{mask_suffix}.PNG")
            Image.fromarray(x0_union_resized_np).save(mask_save_path)
            print(f"Saved mask to: {mask_save_path}")

            fig, axes = plt.subplots(1, 4, figsize=(24, 5))
            axes[0].imshow(original_image_np)
            axes[0].set_title("Input")
            axes[0].axis("off")
            axes[1].imshow(gt_image, cmap="gray")
            axes[1].set_title("GT Mask")
            axes[1].axis("off")
            axes[2].imshow(x0_union_resized_np, cmap="gray")
            axes[2].set_title("Mask")
            axes[2].axis("off")
            axes[3].imshow(skin_applied_result)
            axes[3].set_title("Output")
            axes[3].axis("off")
            generated_suffix = f"mpd_w{mpd_w_str}" if args.method == "mpd" else args.method
            plt.savefig(f"{save_dir}/{filename}_generated_{generated_suffix}.PNG")
            plt.close(fig)

# 儲存 CSV
results_df = pd.DataFrame(results)
csv_suffix = f"mpd_w{mpd_w_str}" if args.method == "mpd" else args.method
csv_output_path = os.path.join(save_dir, f"pigmentation_scores_{csv_suffix}.csv")
results_df.to_csv(csv_output_path, index=False)
print(f"Saved scores to CSV at {csv_output_path}")
