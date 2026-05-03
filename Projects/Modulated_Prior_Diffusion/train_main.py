#train_main
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image


#import shutil
from typing import Tuple
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset


from models.embedding import *
from models.engine import ConditionalGaussianDiffusionTrainer, ConditionalDiffusionEncoderTrainer, DDIMSampler, DDIMSamplerEncoder,ImageConditionalGaussianDiffusionTrainer,DDIMSamplerImage,ImageConditionalGaussianDiffusionTrainer_w,GPCDConcatGaussianDiffusionTrainer,DDIMSamplerImageGPCDConcat 

from dataset_test import CustomImageDatasetCondition_MPD, CustomSampler
from utils import GradualWarmupScheduler, get_model, get_optimizer, get_piecewise_constant_schedule, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, LoadEncoder
#from config import *


try:
    import wandb
except ImportError:
    wandb = None


class NoOpAccelerator:
    def prepare(self, *args):
        return args

    def backward(self, loss):
        loss.backward()


class SmokeImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, img_size):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            "input_image": torch.randn(3, self.img_size, self.img_size),
            "groundtruth_image": torch.randn(3, self.img_size, self.img_size),
            "filename": f"smoke_{index}",
        }


def maybe_limit_dataset(dataset, limit):
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def setup_wandb(args):
    if args.no_wandb or wandb is None:
        if wandb is None and not args.no_wandb:
            print("wandb is not installed; logging disabled.")
        return None
    return wandb


# def generate_ATR2IDX(defects):
#     """
#     根據 defects（瑕疵標籤）生成 ATR2IDX 字典，為每個唯一的 defect 分配一個索引
#     """
#     unique_atr = set(defects)
#     ATR2IDX = {atr: idx for idx, atr in enumerate(unique_atr)}
#     return ATR2IDX

# def generate_OBJ2IDX(images):
#     """
#     根據 defects（瑕疵標籤）生成 ATR2IDX 字典，為每個唯一的 defect 分配一個索引
#     """
#     unique_obj = set(images)
#     OBJ2IDX = {obj: idx for idx, obj in enumerate(unique_obj)}
#     return OBJ2IDX

def main(args):
    
    # initialize W&B
#     wandb.init(
#         project=args.exp,
#         config={
#             "learning_rate": args.lr,
#             "epochs": args.epochs,
#             "dataset": args.csv_file.split('/')[-1],
#             "num_res_blocks": args.num_res_blocks,
#             "img_size": args.img_size,
#             "batch_size": args.batch_size,
#             "ch_mult": args.channel_mult,
#             "guidance scale": args.w
#         },
#         job_type="training"
#     )
    
    wb = setup_wandb(args)
    accelerator = NoOpAccelerator()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
    if args.smoke_test:
        train_ds = SmokeImagePairDataset(args.smoke_samples, args.img_size)
    else:
        train_ds = CustomImageDatasetCondition_MPD(data_root=args.train_data_root, dataset_type="train", transform=transform)
        train_ds = maybe_limit_dataset(train_ds, args.max_train_samples)
        if len(train_ds) == 0:
            raise ValueError(f"No training samples found under {args.train_data_root}")
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #w_values = [0.1 * i for i in range(5, 10)]
    if args.method == "mpd":
        if args.mpd_w_values is not None:
            mpd_w_values = args.mpd_w_values
        else:
            mpd_w_values = [0.5] if args.smoke_test else [round(0.1 * i, 1) for i in range(0, 11)]
    else:
        mpd_w_values = [None]
    
    for mpd_w in mpd_w_values:
    
        # define models
        model = get_model(args)
        trainer = ImageConditionalGaussianDiffusionTrainer_w(model, args.beta, args.num_timestep)
        if args.method == "gpcd_concat":
            trainer = GPCDConcatGaussianDiffusionTrainer(model, args.beta, args.num_timestep)
        # optimizer and learning rate scheduler
        optimizer = get_optimizer(model, args)

        sampler = DDIMSamplerImage(
            model,
            beta=args.beta,
            T=args.num_timestep,
            w=args.w
        )
        if args.method == "gpcd_concat":
            sampler = DDIMSamplerImageGPCDConcat(
                model,
                beta=args.beta,
                T=args.num_timestep,
                w=args.w
            )

        denoiser = model.denoiser if hasattr(model, "denoiser") else model
        denoiser_params = sum(p.numel() for p in denoiser.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"method={args.method} denoiser_params={denoiser_params:,} total_params={total_params:,}")

        if args.encoder_path != None and args.method == "mpd":
            encoder = LoadEncoder(args)
            trainer = ConditionalDiffusionEncoderTrainer(
                encoder = encoder,
                model = model,
                beta = args.beta,
                T = args.num_timestep,
                only_encoder = args.only_encoder
            )

            sampler = DDIMSamplerEncoder(
                model = model,
                encoder = encoder,
                beta = args.beta,
                T = args.num_timestep,
                w = args.w,
                only_encoder = args.only_encoder
            )

        model = model.to(device)
        trainer = trainer.to(device)
        sampler = sampler.to(device)

        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=args.epochs, 
            eta_min=0, 
            last_epoch=-1
        )

        if args.lr_schedule == "cosine":
            warmUpScheduler = GradualWarmupScheduler(
            optimizer=optimizer, 
            multiplier=2.5,
            warm_epoch=max(1, args.epochs // 10), 
            after_scheduler=cosineScheduler
            )
        elif args.lr_schedule == "piecewise":
            warmUpScheduler = get_piecewise_constant_schedule(optimizer, "1:30,0.1:60,0.05")

        elif args.lr_schedule == "linear":
            warmUpScheduler = get_linear_schedule_with_warmup(optimizer, max(1, args.epochs // 10), args.epochs)
        elif args.lr_schedule == "polynomial":
            warmUpScheduler = get_polynomial_decay_schedule_with_warmup(optimizer, max(1, args.epochs // 10), args.epochs)


        dataloader, model, trainer, sampler, optimizer = accelerator.prepare(dataloader, model, trainer, sampler, optimizer)


        for epoch in range(1, args.epochs + 1):

            model.train()
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

            # train
            for batch in progress_bar:
                x = batch["groundtruth_image"].to(device)

                input_images = batch["input_image"].to(device)  # 展開圖像
                B = x.size()[0]

                if args.method == "mpd":
                    loss = trainer(x, input_images, mpd_w=mpd_w).sum() / B ** 2.
                else:
                    loss = trainer(x, input_images).sum() / B ** 2.
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # log training process
                #wandb.log({"loss": loss.item()})

                progress_bar.set_postfix({
                    "loss": f"{loss.item(): .4f}",
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                optimizer.step()
                optimizer.zero_grad()

            #wandb.log({'lr': optimizer.param_groups[0]["lr"]})

            warmUpScheduler.step()

            # validation, save an image of currently generated samples
            size = x.size()[1:]
            if epoch % args.eval_interval == 0:
                eval_epoch = (epoch // args.eval_interval) * args.eval_interval  # 計算最接近的驗證點
                model.eval()
                #print("eval picture size=:",*size)
                # sample random noise
                #if args.onePic_oneCond:
                n_samples = args.eval_batch_size
                if args.smoke_test:
                    valid_ds = SmokeImagePairDataset(args.smoke_samples, args.img_size)
                else:
                    valid_ds = CustomImageDatasetCondition_MPD(data_root=args.eval_data_root, dataset_type="test", transform=transform)
                    valid_ds = maybe_limit_dataset(valid_ds, args.max_eval_samples)
                    if len(valid_ds) == 0:
                        raise ValueError(f"No evaluation samples found under {args.eval_data_root}")
                dataloader_val = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers)
                _ ,x_sample = next(enumerate(dataloader_val))
                x_i = torch.randn(n_samples, *size).to(device)
                c2_image = x_sample["input_image"].to(device)
                if args.method == "mpd":
                    x_i = (1-mpd_w) * x_i + mpd_w * c2_image

                if args.method == "mpd":
                    x0 = sampler(x_i, steps=args.steps)
                else:
                    x0 = sampler(x_i, c2_image, steps=args.steps)
                x0 = x0 * 0.5 + 0.5

                # save image
                x_sample_x_image = x_sample["groundtruth_image"]
                x_sample_x_image = x_sample_x_image * 0.5 + 0.5
                c2_image = c2_image.cpu().detach() * 0.5 + 0.5
                c2_image = c2_image.to(device)
                x_sample_x_image = x_sample_x_image.to(device)
                output = torch.cat( (x_sample_x_image , c2_image , x0 ), axis=-2)

#                 os.makedirs(os.path.join('result', args.exp, f"w_{w:.1f}", args.dir), exist_ok=True)
#                 save_image(output, os.path.join('result', args.exp, args.dir, f'epoch{eval_epoch}_lr{args.lr:.1e}_output_image.png'))


#                 save_root = os.path.join('checkpoint', args.exp, args.dir)
#                 #save_root = os.path.join('checkpoint', args.exp, args.dir)
#                 os.makedirs(save_root, exist_ok=True)

#                 torch.save({
#                     'epoch': epoch,
#                     'model': model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                 }, os.path.join(save_root, f"model_epoch{eval_epoch}_lr{args.lr:.1e}_full.pth"))

                method_dir = f"w_{mpd_w:.1f}_lr{args.lr:.1e}" if args.method == "mpd" else f"{args.method}_lr{args.lr:.1e}"
                result_dir = os.path.join('result', args.exp, method_dir)
                os.makedirs(result_dir, exist_ok=True)
                image_name = f'epoch{eval_epoch}_lr{args.lr:.1e}_w{mpd_w:.1f}_image.png' if args.method == "mpd" else f'epoch{eval_epoch}_lr{args.lr:.1e}_{args.method}_image.png'
                save_image(output, os.path.join(result_dir, image_name))

                # 儲存模型
                save_root = os.path.join('checkpoint', args.exp, method_dir)
                os.makedirs(save_root, exist_ok=True)
                ckpt_name = f"model_epoch{eval_epoch}_lr{args.lr:.1e}_w{mpd_w:.1f}.pth" if args.method == "mpd" else f"model_epoch{eval_epoch}_lr{args.lr:.1e}_{args.method}.pth"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(save_root, ckpt_name))

                





if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    parser = argparse.ArgumentParser()
    #nfs/work/daniel_chou/CDDIM/data/phison
    # General Hyperparameters 
    #parser.add_argument('--arch', type=str, default='unetencoderattention', help='unet architecture')
    #parser.add_argument('--arch', type=str, default='unet', help='unet architecture')
    parser.add_argument('--arch', type=str, default='unetattention_image', help='unet architecture')
    
    parser.add_argument('--eval_batch_size', type=int, default=2, help='evaluate batch size')
    
    #parser.add_argument('--data', type=str, default='/root/notebooks/nfs/work/chiu.li/CDDIM/root/notebooks/nfs/work/daniel_chou/CDDIM/data/phison', help='dataset location')
    #parser.add_argument('--data', type=str, default='/root/notebooks/nfs/work/barry.chen/Phison/Conditional_Diffusion/CDDIM/data/phison', help='dataset location')
    parser.add_argument('--csv_file', type=str, default='/root/notebooks/nfs/work/chiu.li/CDDIM/root/notebooks/nfs/work/daniel_chou/CDDIM/data/PCB_data/Merged/Merged_Final_Info.csv', help='Path to the CSV file')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--eval_interval', type=int, default=100, help='Frequency of evaluation')
    parser.add_argument('--only_table', action='store_true', help="only use embedding table for class embedding")
    parser.add_argument('--fix_emb', action='store_true', help='Freeze embedding table wieght to create fixed label embedding')
    parser.add_argument('--lr_schedule', type=str, default="cosine", choices=["cosine", "piecewise", "linear", "polynomial"])
    
    # ccip parameters
    #parser.add_argument('--encoder_path', type=str, default="/root/notebooks/nfs/work/barry.chen/Phison/Conditional_Diffusion/CDDIM/cMLIP/checkpoint/exp/NoMiss/model_10.pth", help="pretrained weight path of class encoder")
    parser.add_argument('--encoder_path', type=str, default=None, help="pretrained weight path of class encoder")
    parser.add_argument('--only_encoder', action="store_true", help="only use class encoder in ccip model(two tokens)")
    
    # Data hyperparameters
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=64, help='training image size')
    
    # Diffusion hyperparameters
    parser.add_argument('--num_timestep', type=int, default=1000, help='number of timesteps')
    parser.add_argument('--beta', type=Tuple[float, float], default=(0.0001, 0.02), help='beta start, beta end')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--eta', type=float, default=0., help='ddim parameter when sampling')
    parser.add_argument('--exp', type=str, default='MED_pth_w_ISIC', help='experiment directory name')
    parser.add_argument('--dir', type=str, default='MED_image', help='model weight directory')
    parser.add_argument('--sample_method', type=str, default='ddim', choices=['ddpm', 'ddim'], help='sampling method')
    parser.add_argument('--method', type=str, default='mpd', choices=['mpd', 'gpcd_concat'], help='conditioning method')
    parser.add_argument('--steps', type=int, default=100, help='decreased timesteps using ddim')
    parser.add_argument('--drop_prob', type=float, default=0.15, help='probability of dropping label when training diffusion model')
    parser.add_argument('--train_data_root', type=str, default='/workspace/chiu.li/split_ISIC', help='training dataset root')
    parser.add_argument('--eval_data_root', type=str, default='/workspace/chiu.li/split_ISIC', help='evaluation dataset root')
    parser.add_argument('--max_train_samples', type=int, default=None, help='optional limit for training samples')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='optional limit for evaluation samples')
    parser.add_argument('--smoke_test', action='store_true', help='run a tiny synthetic train/eval smoke test')
    parser.add_argument('--smoke_samples', type=int, default=2, choices=[1, 2, 3, 4], help='number of synthetic smoke-test samples')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb logging')
    parser.add_argument('--mpd_w_values', type=float, nargs='+', default=None, help='optional MPD w values; default keeps the original 0.0 to 1.0 sweep')
    
    # UNet hyperparameters
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of residual blocks in unet')
    parser.add_argument('--emb_size', type=int, default=128, help='embedding output dimension')
    parser.add_argument('--w', type=float, default=5, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')  
    parser.add_argument('--concat', action="store_true", help="concat label embedding before CA")
    parser.add_argument('--use_spatial_transformer', action="store_true", help="use transfomer based model to do attention")
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2, 2], help='width of unet model')
   
    # Transformer hyperparameters(Optional)
    parser.add_argument('--projection_dim', type=int, default=512, help='q, k, v dimension in attention layer')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='attention head channels')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads, either specify head_channels or num_heads')
    
    parser.add_argument('--ignored', type=str, nargs='+', default=None, help='exclude folder when loading dataset, for compositional zero-shot generation')
    
    
    args = parser.parse_args()
    
    main(args)
