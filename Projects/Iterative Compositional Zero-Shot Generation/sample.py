# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [64,128, 256 , 512]
       # assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_atr=args.num_condition[0],
        num_obj=args.num_condition[1]
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    #print("state_dict",state_dict)
    
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    #vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c1 =[1]
    c2 = [2]
    # Create sampling noise:
    #n = len(class_labels)
    n =1
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    #y = torch.tensor(class_labels, device=device)
    c1 = torch.tensor(c1, device=device)
    c2 = torch.tensor(c2, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.cat([torch.zeros_like(c1), torch.zeros_like(c2)], dim=0)
    # y = torch.cat([c1, c2, y_null], dim=0)
    #  model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    z = torch.cat([z, z], 0)
    y_null1 = torch.tensor([4] * n, device=device)
    y_null2 = torch.tensor([2] * n, device=device)
    c1 = torch.cat([c1, y_null1], 0)
    c2 = torch.cat([c2, y_null2], 0)
    #y = torch.cat([c1,c2, y_null], 0)

    print(f"Shape of c1: {c1.shape}, Shape of c2: {c2.shape}")

    model_kwargs = dict(c1=c1, c2=c2, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    #samples = (samples + 1) / 2  # 手動反正規化到 [0, 1]


    # Save and display images:
    save_image(samples, "sample2.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[64,128,256, 512], default=128)
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')
    #parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/workspace/DiT/results/Phison3_S3_balance-DiT-XL-2/checkpoints/0035400.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
