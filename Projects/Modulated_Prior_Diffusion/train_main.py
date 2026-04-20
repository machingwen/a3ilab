#MED_train_weight_ISIC
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image


#import shutil
from typing import Tuple
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import wandb
from accelerate import Accelerator


from models.embedding import *
from models.engine import ConditionalGaussianDiffusionTrainer, ConditionalDiffusionEncoderTrainer, DDIMSampler, DDIMSamplerEncoder,ImageConditionalGaussianDiffusionTrainer,DDIMSamplerImage,ImageConditionalGaussianDiffusionTrainer_w 

from dataset_test import CustomImageDatasetCondition_MPD, CustomSampler
from utils import GradualWarmupScheduler, get_model, get_optimizer, get_piecewise_constant_schedule, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, LoadEncoder
from config import *



def generate_ATR2IDX(defects):
    """
    根據 defects（瑕疵標籤）生成 ATR2IDX 字典，為每個唯一的 defect 分配一個索引
    """
    unique_atr = set(defects)
    ATR2IDX = {atr: idx for idx, atr in enumerate(unique_atr)}
    return ATR2IDX

def generate_OBJ2IDX(images):
    """
    根據 defects（瑕疵標籤）生成 ATR2IDX 字典，為每個唯一的 defect 分配一個索引
    """
    unique_obj = set(images)
    OBJ2IDX = {obj: idx for idx, obj in enumerate(unique_obj)}
    return OBJ2IDX

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
    
    accelerator = Accelerator()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
    train_ds = CustomImageDatasetCondition_MPD(data_root="/workspace/chiu.li/split_ISIC", dataset_type="train", transform=transform)
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #w_values = [0.1 * i for i in range(5, 10)]
    mpd_w_values = [round(0.1 * i, 1) for i in range(0, 11)]
    
    for mpd_w in mpd_w_values:
    
        # define models
        model = get_model(args)
        trainer = ImageConditionalGaussianDiffusionTrainer_w(model, args.beta, args.num_timestep)
        # optimizer and learning rate scheduler
        optimizer = get_optimizer(model, args)

        sampler = DDIMSamplerImage(
            model,
            beta=args.beta,
            T=args.num_timestep,
            w=args.w
        )

        if args.encoder_path != None:
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
            warm_epoch=args.epochs // 10, 
            after_scheduler=cosineScheduler
            )
        elif args.lr_schedule == "piecewise":
            warmUpScheduler = get_piecewise_constant_schedule(optimizer, "1:30,0.1:60,0.05")

        elif args.lr_schedule == "linear":
            warmUpScheduler = get_linear_schedule_with_warmup(optimizer, args.epochs // 10, args.epochs)
        elif args.lr_schedule == "polynomial":
            warmUpScheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.epochs // 10, args.epochs)


        dataloader, model, trainer, sampler, optimizer = accelerator.prepare(dataloader, model, trainer, sampler, optimizer)


        for epoch in range(1, args.epochs + 1):

            model.train()
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

            # train
            for batch in progress_bar:
                x = batch["groundtruth_image"].to(device)

                input_images = batch["input_image"].to(device)  # 展開圖像
                B = x.size()[0]

                loss = trainer(x, input_images, mpd_w=mpd_w).sum() / B ** 2.
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
                valid_ds  =CustomImageDatasetCondition_MPD(data_root="/workspace/chiu.li/split_ISIC", dataset_type="test", transform=transform)
                dataloader_val = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers)
                _ ,x_sample = next(enumerate(dataloader_val))
                x_i = torch.randn(n_samples, *size).to(device)
                c2_image = x_sample["input_image"].to(device)
                x_i = (1-mpd_w) * x_i + mpd_w * c2_image

                x0 = sampler(x_i, steps=args.steps)
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

                result_dir = os.path.join('result', args.exp, f"w_{mpd_w:.1f}_lr{args.lr:.1e}")
                os.makedirs(result_dir, exist_ok=True)
                save_image(output, os.path.join(result_dir, f'epoch{eval_epoch}_lr{args.lr:.1e}_w{mpd_w:.1f}_image.png'))

                # 儲存模型
                save_root = os.path.join('checkpoint', args.exp, f"w_{mpd_w:.1f}_lr{args.lr:.1e}")
                os.makedirs(save_root, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(save_root, f"model_epoch{eval_epoch}_lr{args.lr:.1e}_w{mpd_w:.1f}.pth"))

                





if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    os.system("rm -rf /root/notebooks/nfs/work/daniel_chou/CDDIM/data/phison/.ipynb_checkpoints")
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
    parser.add_argument('--steps', type=int, default=100, help='decreased timesteps using ddim')
    parser.add_argument('--drop_prob', type=float, default=0.15, help='probability of dropping label when training diffusion model')
    
    # UNet hyperparameters
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of residual blocks in unet')
    parser.add_argument('--emb_size', type=int, default=128, help='embedding output dimension')
    parser.add_argument('--w', type=float, default=5, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')  
    parser.add_argument('--concat', action="store_true", help="concat label embedding before CA")
    parser.add_argument('--use_spatial_transformer', action="store_true", help="use transfomer based model to do attention")
    parser.add_argument('--channel_mult', type=list, default=[1, 2, 2, 2], help='width of unet model')
   
    # Transformer hyperparameters(Optional)
    parser.add_argument('--projection_dim', type=int, default=512, help='q, k, v dimension in attention layer')
    parser.add_argument('--num_head_channels', type=int, default=-1, help='attention head channels')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads, either specify head_channels or num_heads')
    
    parser.add_argument('--ignored', type=str, nargs='+', default=None, help='exclude folder when loading dataset, for compositional zero-shot generation')
    
    
    args = parser.parse_args()
    
    main(args)
