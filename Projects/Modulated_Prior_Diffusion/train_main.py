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
import csv
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import random
import time
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng_state(state):
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_validation_rng(seed, device):
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def setup_wandb(args):
    if args.no_wandb or wandb is None:
        if wandb is None and not args.no_wandb:
            print("wandb is not installed; logging disabled.")
        return None
    project = args.wandb_project or args.exp
    run_name = args.wandb_run_name
    if run_name is None:
        if args.method == "mpd":
            run_name = f"{args.exp}-{args.method}-w{args.w:g}"
        else:
            run_name = f"{args.exp}-{args.method}"
    wandb.init(
        project=project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        job_type="training",
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("val/*", step_metric="train/global_step")
    return wandb


def get_mpd_w_values(args):
    if args.method != "mpd":
        return [None]
    if args.sweep_mpd_w:
        return args.mpd_w_values or [round(0.1 * i, 1) for i in range(0, 11)]
    if args.mpd_w_values is not None:
        if len(args.mpd_w_values) != 1:
            raise ValueError("Default publication mode runs one MPD job. Use --sweep_mpd_w for multiple values.")
        return args.mpd_w_values
    return [args.w]


def mask_metrics(pred, target, threshold=0.5, eps=1e-7):
    pred_mask = (pred.mean(dim=1) > threshold)
    target_mask = (target.mean(dim=1) > threshold)
    intersection = (pred_mask & target_mask).sum(dim=(1, 2)).float()
    union = (pred_mask | target_mask).sum(dim=(1, 2)).float()
    pred_sum = pred_mask.sum(dim=(1, 2)).float()
    target_sum = target_mask.sum(dim=(1, 2)).float()
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    return iou, dice


def append_metrics_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "phase",
        "epoch",
        "method",
        "mpd_w",
        "steps",
        "train_loss",
        "val_loss",
        "iou",
        "dice",
        "inference_time_per_image_sec",
        "eval_samples",
        "checkpoint_path",
        "total_inference_time_sec",
        "sampler_calls",
        "denoiser_forward_calls",
        "denoiser_calls_per_image",
        "avg_time_per_denoiser_forward_sec",
        "sampler",
    ]
    rows = None
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fieldnames:
                rows = list(reader)
                write_header = True
    mode = "w" if rows is not None else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        if rows is not None:
            for old_row in rows:
                writer.writerow({key: old_row.get(key, "") for key in fieldnames})
        writer.writerow(row)


def normalize_state_dict_keys(state_dict):
    new_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_dict[key[7:]] = value
        else:
            new_dict[key] = value
    return new_dict


def resume_training_state(args, checkpoint_path, model, optimizer, scheduler, device, num_batches):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(normalize_state_dict_keys(model_state))

    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    else:
        print(f"resume checkpoint has no optimizer state: {checkpoint_path}")

    completed_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    if completed_epoch > 0:
        scheduler.step(completed_epoch)

    global_step = int(checkpoint.get("global_step", completed_epoch * num_batches)) if isinstance(checkpoint, dict) else completed_epoch * num_batches
    start_epoch = completed_epoch + 1
    if start_epoch > args.epochs:
        raise ValueError(
            f"resume checkpoint is already at epoch {completed_epoch}, "
            f"but --epochs is {args.epochs}. Set --epochs higher to extend training."
        )
    print(f"resumed from {checkpoint_path}: start_epoch={start_epoch}, target_epochs={args.epochs}, global_step={global_step}")
    return start_epoch, global_step


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

    mpd_w_values = get_mpd_w_values(args)
    global_step = 0
    if args.resume_checkpoint is not None and len(mpd_w_values) != 1:
        raise ValueError("--resume_checkpoint can only be used with one MPD w value. Disable --sweep_mpd_w or pass one --mpd_w_values value.")
    
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

        start_epoch = 1
        if args.resume_checkpoint is not None:
            start_epoch, global_step = resume_training_state(
                args=args,
                checkpoint_path=args.resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=warmUpScheduler,
                device=device,
                num_batches=len(dataloader),
            )

        dataloader, model, trainer, sampler, optimizer = accelerator.prepare(dataloader, model, trainer, sampler, optimizer)


        for epoch in range(start_epoch, args.epochs + 1):

            model.train()
            trainer.train()
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
            epoch_losses = []

            # train
            for batch in progress_bar:
                x = batch["groundtruth_image"].to(device)

                input_images = batch["input_image"].to(device)  # 展開圖像
                B = x.size()[0]
                micro_batch_size = args.micro_batch_size or B
                batch_loss = 0.0
                for start in range(0, B, micro_batch_size):
                    end = min(start + micro_batch_size, B)
                    x_micro = x[start:end]
                    input_micro = input_images[start:end]
                    if args.method == "mpd":
                        loss = trainer(x_micro, input_micro, mpd_w=mpd_w).sum() / B ** 2.
                    else:
                        loss = trainer(x_micro, input_micro).sum() / B ** 2.
                    accelerator.backward(loss)
                    batch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # log training process
                #wandb.log({"loss": loss.item()})

                progress_bar.set_postfix({
                    "loss": f"{batch_loss: .4f}",
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                epoch_losses.append(batch_loss)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if wb is not None:
                    log_data = {
                        "train/global_step": global_step,
                        "train/loss": batch_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/micro_batch_size": micro_batch_size,
                    }
                    if mpd_w is not None:
                        log_data["train/mpd_w"] = mpd_w
                    wb.log(log_data, step=global_step)

            #wandb.log({'lr': optimizer.param_groups[0]["lr"]})

            warmUpScheduler.step()

            # validation, save an image of currently generated samples
            size = x.size()[1:]
            if epoch % args.eval_interval == 0:
                eval_epoch = (epoch // args.eval_interval) * args.eval_interval  # 計算最接近的驗證點
                validation_rng_state = capture_rng_state(device) if args.val_seed is not None else None
                val_generator = seed_validation_rng(args.val_seed, device)
                model.eval()
                trainer.eval()
                if args.smoke_test:
                    valid_ds = SmokeImagePairDataset(args.smoke_samples, args.img_size)
                else:
                    valid_ds = CustomImageDatasetCondition_MPD(data_root=args.eval_data_root, dataset_type="test", transform=transform)
                    valid_ds = maybe_limit_dataset(valid_ds, args.max_eval_samples)
                    if len(valid_ds) == 0:
                        raise ValueError(f"No evaluation samples found under {args.eval_data_root}")
                dataloader_val = DataLoader(
                    valid_ds,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    generator=val_generator,
                    worker_init_fn=seed_worker if args.val_seed is not None else None,
                )

                iou_sum = 0.0
                dice_sum = 0.0
                eval_count = 0
                inference_seconds = 0.0
                val_losses = []
                output = None

                with torch.no_grad():
                    for x_sample in dataloader_val:
                        c2_image = x_sample["input_image"].to(device)
                        x_sample_x_image = x_sample["groundtruth_image"].to(device)
                        batch_size = c2_image.shape[0]
                        micro_batch_size = args.micro_batch_size or batch_size
                        val_batch_loss = 0.0
                        for start in range(0, batch_size, micro_batch_size):
                            end = min(start + micro_batch_size, batch_size)
                            target_micro = x_sample_x_image[start:end]
                            cond_micro = c2_image[start:end]
                            if args.method == "mpd":
                                val_loss = trainer(target_micro, cond_micro, mpd_w=mpd_w).sum() / batch_size ** 2.
                            else:
                                val_loss = trainer(target_micro, cond_micro).sum() / batch_size ** 2.
                            val_batch_loss += val_loss.item()
                        val_losses.append(val_batch_loss)

                        x_i = torch.randn(batch_size, *size).to(device)
                        if args.method == "mpd":
                            x_i = (1 - mpd_w) * x_i + mpd_w * c2_image

                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        if args.method == "mpd":
                            x0 = sampler(x_i, steps=args.steps)
                        else:
                            x0 = sampler(x_i, c2_image, steps=args.steps)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        inference_seconds += time.perf_counter() - start_time

                        x0 = torch.clamp(x0 * 0.5 + 0.5, 0.0, 1.0)
                        target = torch.clamp(x_sample_x_image * 0.5 + 0.5, 0.0, 1.0)
                        iou, dice = mask_metrics(x0, target)
                        iou_sum += iou.sum().item()
                        dice_sum += dice.sum().item()
                        eval_count += batch_size

                        if output is None:
                            ref = torch.clamp(c2_image * 0.5 + 0.5, 0.0, 1.0)
                            output = torch.cat((target, ref, x0), axis=-2)

                mean_iou = iou_sum / max(1, eval_count)
                mean_dice = dice_sum / max(1, eval_count)
                inference_time_per_image = inference_seconds / max(1, eval_count)
                train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                valid_loss = float(np.mean(val_losses)) if val_losses else float("nan")

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
                if output is not None:
                    save_image(output, os.path.join(result_dir, image_name))

                metrics_path = os.path.join(result_dir, "metrics.csv")
                append_metrics_csv(metrics_path, {
                    "phase": "train_eval",
                    "epoch": eval_epoch,
                    "method": args.method,
                    "mpd_w": "" if mpd_w is None else f"{mpd_w:.6g}",
                    "steps": args.steps,
                    "train_loss": f"{train_loss:.6g}",
                    "val_loss": f"{valid_loss:.6g}",
                    "iou": f"{mean_iou:.6g}",
                    "dice": f"{mean_dice:.6g}",
                    "inference_time_per_image_sec": f"{inference_time_per_image:.6g}",
                    "eval_samples": eval_count,
                    "checkpoint_path": "",
                })
                print(
                    f"validation epoch={eval_epoch} method={args.method} "
                    f"mpd_w={mpd_w if mpd_w is not None else 'n/a'} "
                    f"train_loss={train_loss:.6g} val_loss={valid_loss:.6g} "
                    f"iou={mean_iou:.6g} "
                    f"dice={mean_dice:.6g} "
                    f"inference_time_per_image_sec={inference_time_per_image:.6g} "
                    f"metrics_csv={metrics_path}"
                )
                if wb is not None:
                    log_data = {
                        "train/global_step": global_step,
                        "val/loss": valid_loss,
                        "val/iou": mean_iou,
                        "val/dice": mean_dice,
                        "val/inference_time_per_image_sec": inference_time_per_image,
                        "val/eval_samples": eval_count,
                        "val/epoch": eval_epoch,
                        "val/steps": args.steps,
                    }
                    if mpd_w is not None:
                        log_data["val/mpd_w"] = mpd_w
                    wb.log(log_data, step=global_step)

                # 儲存模型
                save_root = os.path.join('checkpoint', args.exp, method_dir)
                os.makedirs(save_root, exist_ok=True)
                ckpt_name = f"model_epoch{eval_epoch}_lr{args.lr:.1e}_w{mpd_w:.1f}.pth" if args.method == "mpd" else f"model_epoch{eval_epoch}_lr{args.lr:.1e}_{args.method}.pth"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler_epoch': epoch,
                    'global_step': global_step,
                }, os.path.join(save_root, ckpt_name))
                restore_rng_state(validation_rng_state)

                

    if wb is not None:
        wb.finish()





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
    parser.add_argument('--micro_batch_size', type=int, default=None, help='optional backward microbatch size for memory-limited GPUs')
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
    parser.add_argument('--val_seed', type=int, default=None, help='optional fixed seed for each validation epoch')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='checkpoint path to resume training from; set --epochs to the new total epoch count')
    parser.add_argument('--smoke_test', action='store_true', help='run a tiny synthetic train/eval smoke test')
    parser.add_argument('--smoke_samples', type=int, default=2, choices=[1, 2, 3, 4], help='number of synthetic smoke-test samples')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name; defaults to --exp')
    parser.add_argument('--wandb_entity', type=str, default=None, help='optional W&B entity/team')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='optional W&B run name')
    parser.add_argument('--mpd_w_values', type=float, nargs='+', default=None, help='MPD w values. In default single-run mode, provide at most one value; use --sweep_mpd_w for multiple values.')
    parser.add_argument('--sweep_mpd_w', action='store_true', help='run the legacy MPD w sweep; defaults to 0.0, 0.1, ..., 1.0 unless --mpd_w_values is provided')
    
    # UNet hyperparameters
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of residual blocks in unet')
    parser.add_argument('--emb_size', type=int, default=128, help='embedding output dimension')
    parser.add_argument('--w', type=float, default=0.9, help='single-run MPD condition strength; also used as sampler guidance where supported')
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
