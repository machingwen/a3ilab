import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms



from typing import Tuple
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import wandb
from accelerate import Accelerator

from models.embedding import *
from models.ccip import CCIPModel
from dataset import CustomImageDataset, CustomSampler
from utils import GradualWarmupScheduler, get_model
from config import *


def main(args):
    
    # initialize W&B
    wandb.init(
        project=args.exp,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "dataset": args.data.split('/')[-1],
            "img_size": args.img_size,
            "batch_size": args.batch_size,
        },
        job_type="training"
    )
    
    accelerator = Accelerator()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_ds = CustomImageDataset(root=args.data, transform=transform, ignored=args.ignored)
    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_ds = CustomImageDataset(
        root="/root/notebooks/nfs/work/dataset/toy_dataset_66_500",
        transform=transform,
        ignored=args.ignored
    )
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     sampler = CustomSampler(train_ds)
#     dataloader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
#     dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    args.num_condition[0] = len(ATR2IDX)
    args.num_condition[1] = len(OBJ2IDX)
    
    # define models
    model = CCIPModel(
        num_atr = args.num_condition[0],
        num_obj = args.num_condition[1],
        class_embedding = args.emb_dim
    ).to(device)
    
    
    # optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
#     cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer=optimizer, 
#         T_max=args.epochs, 
#         eta_min=0, 
#         last_epoch=-1
#     )
    
#     warmUpScheduler = GradualWarmupScheduler(
#         optimizer=optimizer, 
#         multiplier=2.5,
#         warm_epoch=args.epochs // 10, 
#         after_scheduler=cosineScheduler
#     )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    
    
    dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
    
    for epoch in range(1, args.epochs + 1):
        
        model.train()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        train_loss = 0
        val_loss = 0
        # train
        for batch in progress_bar:
            x = batch["image"].to(device)
            c1 = [ATR2IDX[a] for a in batch["atr"]]
            c2 = [OBJ2IDX[o] for o in batch["obj"]]
            c1 = torch.tensor(c1, dtype=torch.long, device=device)
            c2 = torch.tensor(c2, dtype=torch.long, device=device)
            B = x.size()[0]
            
            loss = model(x, atr=c1, obj=c2)
            train_loss += loss.item()
            optimizer.zero_grad()
            accelerator.backward(loss)
#             loss.backward()
            optimizer.step()
            
                        
            # log training process
            wandb.log({"Train loss": loss.item()})
            
            progress_bar.set_postfix({
                "loss": f"{loss.item(): .4f}",
                "lr": optimizer.state_dict()['param_groups'][0]["lr"]
            })
            
        print(f"Epoch {epoch} Train avg loss: {train_loss / len(dataloader): .4f}")
        train_loss = 0
        
        # validation, save an image of currently generated samples
        model.eval()
        progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        with torch.no_grad():
            
            # sample random noise
            for batch in progress_bar:
                x = batch["image"].to(device)
                c1 = [ATR2IDX[a] for a in batch["atr"]]
                c2 = [OBJ2IDX[o] for o in batch["obj"]]
                c1 = torch.tensor(c1, dtype=torch.long, device=device)
                c2 = torch.tensor(c2, dtype=torch.long, device=device)
                B = x.size()[0]
            
                loss = model(x, atr=c1, obj=c2)
                val_loss += loss.item()
                # log training process
                wandb.log({"Val loss": loss.item()})

                progress_bar.set_postfix({
                    "loss": f"{loss.item(): .4f}",
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        
        print(f"Validation Epoch {epoch} Val avg loss: {val_loss / len(val_loader): .4f}")
        lr_scheduler.step(val_loss / len(val_loader))
        val_loss = 0
                          
        # save model
        save_root = os.path.join('checkpoints', args.exp)
        os.makedirs(save_root, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(save_root, f"model_{epoch}.pth"))
        
        
                





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # General Hyperparameters 
    parser.add_argument('--data', type=str, default='/root/notebooks/nfs/work/dataset/conditional_ut', help='dataset location')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay coefficient')
    parser.add_argument('--emb_dim', type=int, default=512, help='Dimension of class embedding')
    
    # Data hyperparameters
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=128, help='training image size')
    parser.add_argument('--exp', type=str, default='exp', help='experiment directory name')
    parser.add_argument('--num_condition', type=int, nargs="+", help='number of classes in each condition')
    
    parser.add_argument('--ignored', type=str, nargs='+', default=None, help='exclude folder when loading dataset, for compositional zero-shot generation')
    args = parser.parse_args()
    
    main(args)