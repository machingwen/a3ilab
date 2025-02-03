#  Torch imports
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import random
import numpy as np
import sys

from networks.mobilenetv3_regroup import SupConMobileNetV3Large
from lib.datasets_mo import get_dataset
from util_mo import *
import time

sys.path.append('/root/notebooks/PCB_dataset/clustimage_phison/clustimage/')
from clustimage import Clustimage


def set_model(args):
    model = SupConMobileNetV3Large()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion_com = torch.nn.CrossEntropyLoss().cuda()
    
    params_com = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "cls_out" not in key:
                if "cls_classifier" not in key:
                    params_com += [{'params': [value], 'lr': args.learning_rate, 'weight_decay': args.weight_decay}]
        
    optimizer_com = torch.optim.SGD(
        params_com,
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    milestones = [20, 30, 40]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_com, milestones=milestones, gamma=0.1
    )
        
    return model , criterion_com , optimizer_com


def main(args):
    
    args.output_dir = './output/train_regroup/'
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model , criterion_com , optimizer_com = set_model(args)
    
    ds = get_dataset(args.dataset,args.seed, root=args.data_root)
    train_com_loader, train_regroup_df ,train_regroup_loader ,df = ds
    
    embeddings_train, labels_train, _, name_list_train, _, _ = get_features_weight(model, train_regroup_loader)
    clust = Clustimage(method='pca')
    results = clust.fit_transform(embeddings_train, name_list_train, min_clust=10, max_clust=20)
    clust.save(f'clust/{args.seed}_pretrain_all_clustimage_model', overwrite=True)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to use for training"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")

    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")

    parser.add_argument(
        "--dataset",
        default="PHISON_df",
        choices=["CIFAR10", "CIFAR100", "PHISON","PHISON_ori","PHISON_df"],
        help="Pick a dataset",
    )

    parser.add_argument(
        "--kernel",
        default="RBF",
        choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
        help="Pick a kernel",
    )

    parser.add_argument(
        "--no_spectral_conv",
        action="store_false",
        dest="spectral_conv",
        help="Don't use spectral normalization on the convolutions",
    )

    parser.add_argument(
        "--no_spectral_bn",
        action="store_false",
        dest="spectral_bn",
        help="Don't use spectral normalization on the batch normalization layers",
    )

    parser.add_argument(
        "--sngp",
        action="store_true",
        help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)",
    )
    parser.add_argument(
        "--likelihood",
        action="store_true",
        help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)",
    )
    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )
    parser.add_argument(
        "--n_inducing_points_cls", type=int, help="Number of inducing points"
    )

    parser.add_argument("--seed", type=int, default=1 , help="Seed to use for training")

    parser.add_argument(
        "--coeff", type=float, default=3, help="Spectral normalization coefficient"
    )

    parser.add_argument(
        "--n_power_iterations", default=1, type=int, help="Number of power iterations"
    )

    parser.add_argument(
        "--output_dir", default="./default", type=str, help="Specify output directory"
    )
    parser.add_argument(
        "--data_root", default="./data", type=str, help="Specify data directory"
    )

    args = parser.parse_args()
    
    
    try:
        main(args)
    except KeyboardInterrupt:
        print('error')
