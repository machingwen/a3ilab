import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from collections import OrderedDict
from typing import Tuple
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import wandb
from accelerate import Accelerator

from models.embedding import *
from models.engine import ConditionalGaussianDiffusionTrainer, ConditionalDiffusionEncoderTrainer, DDIMSampler, DDIMSamplerEncoder 
from dataset import CustomImageDataset
from utils import GradualWarmupScheduler, get_model, get_optimizer, get_piecewise_constant_schedule, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, LoadEncoder
from config import *
from models.unet import UNetEncoderAttention


def main(args):
    
    accelerator = Accelerator()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetEncoderAttention(
            T=1000,
            image_size=64,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=[8,4,2],
            dropout=0.15,
            channel_mult=args.channel_mult,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            use_spatial_transformer=args.use_spatial_transformer,
            context_dim=args.context_dim,
        )
    model.eval()
          
        # validation, save an image of currently generated samples
    ckpt = torch.load(os.path.join("/home/barry/NYCU/Conditional_Diffusion/checkpoints/phison_cmlip/unseen_brokeF1210_shiftC0604/","model_100.pth"))

    model.load_state_dict(ckpt)            
    # model.load_state_dict(ckpt)
    print("model load weight done.") 
 
    sampler = DDIMSamplerEncoder(
            model = model,
            encoder = LoadEncoder(args),
            beta = args.beta,
            T = args.num_timestep,
            w = args.w,
            only_encoder = args.only_encoder
        )
    
    model, sampler = accelerator.prepare(model, sampler)  
    # sample random noise
    x_i =torch.randn(64, 3, 64, 64).to(device)
    # create conditions of each class
    # create conditions like [0,0,0,1,1,1, ...] [0,1,2,3,0,1,2,3, ...]
    Condition = torch.tensor([1]).to(device)
    Condition = Condition.repeat(6, 1).permute(1, 0).reshape(-1)
    c1 = Condition.repeat(100)[:64].long().to(device)
    labels = torch.tensor([2]).to(device)
    c2 = labels.repeat(100)[:64].long().to(device)
    print("Condition: ", c1)
    print("labels: ", c2)

    x0 = sampler(x_i, c1, c2, steps=args.steps)
    
    # save image
    os.makedirs(os.path.join('result', args.exp, args.dir), exist_ok=True)
    save_image(x0, os.path.join('result', args.exp, args.dir, f'eva.png'))

                





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # General Hyperparameters 
    parser.add_argument('--arch', type=str, default='unetencoderattention', help='unet architecture')
    parser.add_argument('--data', type=str, default='/home/barry/Conditional_Diffusion/data/real_phison', help='dataset location')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--eval_interval', type=int, default=10, help='Frequency of evaluation')
    parser.add_argument('--only_table', action='store_true', help="only use embedding table for class embedding")
    parser.add_argument('--fix_emb', action='store_true', help='Freeze embedding table wieght to create fixed label embedding')
    parser.add_argument('--lr_schedule', type=str, default="cosine", choices=["cosine", "piecewise", "linear", "polynomial"])
    
    # ccip parameters
    parser.add_argument('--encoder_path', type=str, default="/home/barry/Conditional_Diffusion/checkpoints/exp/NoMiss/model_10.pth", help="pretrained weight path of class encoder")
    parser.add_argument('--only_encoder', action="store_true", help="only use class encoder in ccip model(two tokens)")
    
    # Data hyperparameters
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=64, help='training image size')
    
    # Diffusion hyperparameters
    parser.add_argument('--num_timestep', type=int, default=1000, help='number of timesteps')
    parser.add_argument('--beta', type=Tuple[float, float], default=(0.0001, 0.02), help='beta start, beta end')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--eta', type=float, default=0., help='ddim parameter when sampling')
    parser.add_argument('--exp', type=str, default='phison_cmlip', help='experiment directory name')
    parser.add_argument('--dir', type=str, default='NoMiss', help='model weight directory')
    parser.add_argument('--sample_method', type=str, default='ddim', choices=['ddpm', 'ddim'], help='sampling method')
    parser.add_argument('--steps', type=int, default=100, help='decreased timesteps using ddim')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='probability of dropping label when training diffusion model')
    
    # UNet hyperparameters
    parser.add_argument('--num_res_blocks', type=int, default=3, help='number of residual blocks in unet')
    parser.add_argument('--emb_size', type=int, default=128, help='embedding output dimension')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')
    parser.add_argument('--concat', action="store_true", help="concat label embedding before CA")
    parser.add_argument('--use_spatial_transformer', action="store_true", help="use transfomer based model to do attention")
    
    # Transformer hyperparameters(Optional)
    parser.add_argument('--context_dim', type=int, default=256, help='q, k, v dimension in attention layer')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='attention head channels')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads, either specify head_channels or num_heads')
    parser.add_argument('--channel_mult', type=list, default=[1, 2, 2, 2], help='width of unet model')
    parser.add_argument('--ignored', type=str, nargs='+', default=None, help='exclude folder when loading dataset, for compositional zero-shot generation')
    
    
    args = parser.parse_args()
    
    main(args)