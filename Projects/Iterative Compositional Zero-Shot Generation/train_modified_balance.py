#################################################################################
#                                  Training Loop                                #
#################################################################################
#torchrun --nnodes=1 --nproc_per_node=1 train_modifiedbalance.py --model DiT-XL/2 --num_condition 4 2
"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
#from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataset import CustomImageDataset

import numpy as np
from collections import OrderedDict, Counter
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
from tqdm import tqdm
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from config import *

from torch.utils.data import Sampler

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class BalancedDistributedSampler(Sampler):
    """
    Distributed Sampler with Exponential Smoothing:
    w_i ∝ 1 / (N_i ** gamma)
    Smaller gamma  → less aggressive over-sampling of rare classes.
    Handles multi-label classes by converting them to tuples.
    """
    def __init__(self,
                 dataset,
                 labels,                       # list/array of class indices for each sample
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=42,
                 gamma=0.7):                   # smoothing parameter (0~1)
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        # Store labels as a numpy array for efficient indexing later
        self.labels = np.array(labels)
        self.gamma   = gamma
        self.shuffle = shuffle
        self.seed    = seed
        self.epoch   = 0
        self.num_replicas = num_replicas
        self.rank         = rank
        
        # NEW: Initialize a dictionary to store sampling statistics for each epoch
        self.sampled_class_counts = {}

        # --- Step 1: Count samples for each class ---
        # FIX: Convert unhashable labels (like lists/arrays) to hashable tuples for Counter
        hashable_labels = [tuple(l) for l in self.labels]
        class_cnt = Counter(hashable_labels)

        # --- Step 2: Calculate smoothed weights w_i = 1 / N_i^γ ---
        # Use the same hashable labels to look up the counts
        self.weights = np.array([1.0 / (class_cnt[l] ** self.gamma) for l in hashable_labels], dtype=np.float64)

        # --- Step 3: Normalize to get sampling probabilities (important!) ---
        self.weights = self.weights / self.weights.sum()

        # How many samples each replica should draw
        self.num_samples = int(np.ceil(len(self.dataset) / self.num_replicas))
                     
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Generate the total list of indices for all replicas
        # This sampling result is identical across all ranks because (seed + epoch) is the same
        all_indices = rng.choice(
            len(self.dataset),
            size=self.num_samples * self.num_replicas,
            replace=True,
            p=self.weights,
            shuffle=False
        )

        # NEW: Calculate and store the count of each sampled class
        # This step is done only on rank 0 to avoid redundant computation and I/O
        if self.rank == 0:
            # Get the corresponding labels via all_indices
            sampled_labels = self.labels[all_indices]
            # FIX: Convert the sampled labels (which are numpy arrays) to hashable tuples before counting
            hashable_sampled_labels = [tuple(l) for l in sampled_labels]
            # Use Counter to count them
            self.sampled_class_counts = Counter(hashable_sampled_labels)

        # Each rank takes its own slice of indices (sharding)
        indices = all_indices[self.rank:len(all_indices):self.num_replicas]

        # For added randomness, shuffle the indices within each rank
        if self.shuffle:
            # Each rank uses a different seed for shuffling to ensure their order is different
            rank_rng = np.random.default_rng(self.seed + self.epoch + self.rank)
            rank_rng.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        # Reset the counter for the new epoch's statistics
        self.sampled_class_counts = {}

    def get_sampled_class_counts(self):
        """
        Returns the class counts sampled across all replicas in the last __iter__ call.
        Note: This is the total count before sharding the data to individual GPUs.
        """
        return self.sampled_class_counts
    
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/CelebA-modifiedbalance-itr2-gamma0.7-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_atr=args.num_condition[0],
        num_obj=args.num_condition[1]
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="", rescale_learned_sigmas=True, learn_sigma=True)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Setup data:
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    ds_type = args.data_path.split("/")[-1]
    if "ut" in ds_type:
        CFG = Zappo50K()
    elif "CelebA" in ds_type:
        CFG = CelebA()
    elif "Phison" in ds_type:
        CFG = Phison()
    elif "Mnist" in ds_type:
        CFG = Mnist()
    else:
        CFG = toy_dataset()
        
    ATR2IDX = CFG.ATR2IDX
    OBJ2IDX = CFG.OBJ2IDX
    
    train_ds = CustomImageDataset(root=args.data_path, transform=transform, ignored=None)
    
    # 使用我們修復好的 Sampler
    sampler = BalancedDistributedSampler(
        dataset=train_ds,
        labels=train_ds.labels,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
        gamma=0.5
    )

    dataloader = DataLoader(
        train_ds, 
        batch_size=int(args.global_batch_size // dist.get_world_size()), 
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )
    
    logger.info(f"Dataset contains {len(train_ds):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Variables for monitoring/logging purposes:
    train_steps = 0 # 總步數計數器保持不變

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        # 記錄類別分佈
        if rank == 0:
            _ = iter(dataloader)
            class_counts = sampler.get_sampled_class_counts()
            if class_counts:
                # 這裡的 tuple(k) 是為了處理多條件標籤，確保可以排序
                counts_str = ", ".join([f"Class {tuple(k)}: {v}" for k, v in sorted(class_counts.items())])
                logger.info(f"Epoch {epoch} Sampled Distribution: {counts_str}")

        # MODIFIED: 重置每個 epoch 的監控變數
        running_loss = 0
        log_steps = 0
        start_time = time()

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(rank != 0))
        for batch in progress_bar:
            x = batch["image"].to(device)
            c1 = [ATR2IDX[a] for a in batch["atr"]]
            c2 = [OBJ2IDX[o] for o in batch["obj"]]
            c1 = torch.tensor(c1, dtype=torch.long, device=device)
            c2 = torch.tensor(c2, dtype=torch.long, device=device)
            
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(c1=c1, c2=c2)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # 累加損失值
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            # REMOVED: 移除了基於 `log_every` 的日誌記錄區塊

        # ---- END OF EPOCH ----

        # MODIFIED: 在 epoch 結束後計算並記錄平均損失
        torch.cuda.synchronize()
        epoch_duration = time() - start_time
        
        # 計算並在所有進程間同步平均損失
        avg_loss = torch.tensor(running_loss / log_steps, device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()

        if rank == 0:
            logger.info(f"Epoch {epoch} Summary | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")

        # 儲存 Checkpoint
        if epoch % args.ckpt_every == 0:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch:04d}-steps-{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/workspace/DiT/CelebA_itr2_modified_balance", help="Path to the training dataset")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=128)
    parser.add_argument('--num_condition', type=int, nargs='+', help='number of classes in each condition')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--global-batch-size", type=int, default=120)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=50, help="Save a checkpoint every N epochs")
    args = parser.parse_args()
    main(args)