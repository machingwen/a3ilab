# -*- coding: utf-8 -*-

# https://github.com/QuocThangNguyen/deep-metric-learning-tsinghua-dogs/blob/master/src/scripts/visualize_tsne.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score

import argparse
import os
from multiprocessing import cpu_count
import time
from pprint import pformat
import logging
import sys
from typing import Dict, Any, List, Tuple
# from networks.mobilenetv3_HybridExpert import SupConMobileNetV3Large
from util_mo import *
import torch.backends.cudnn as cudnn
from due import dkl_Phison_mo, dkl_Phison_mo_s2
# from due.wide_resnet_Phison_old import WideResNet
from lib.datasets_mo import get_dataset
from gpytorch.likelihoods import SoftmaxLikelihood
import gpytorch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# plt.rcParams['figure.figsize'] = (32, 32)
# plt.rcParams['figure.dpi'] = 150

def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_uncertainty(model_s1, model,likelihood, dataloader, tsne=False):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model_s1.eval()
    model.eval()
    model.to(device)
    likelihood.to(device)
    # we'll store the features as NumPy array of size num_images x feature_size
    uncertainty = None
    
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    name_list = []
    full_name_list = []
    tsne_label_list = []

    print("Start calculate uncertainty")
    for i, data in enumerate(tqdm(dataloader)):

        try:
            img, target, file_path, name, full_name = data
        except:
            img, target, file_path, name = data

#         feat_list = []
#         def hook(module, input, output):
#             feat_list.append(output.clone().detach())

        images = img.to(device)
        target = target.squeeze().tolist()

        if isinstance(target, str):
            labels.append(target)
        else:
            if isinstance(target, int):
                labels.append(target)
            else:
                for p in file_path:
                    image_paths.append(p)
                for lb in target:
                    labels.append(lb)        
        try:
            if name is not None:
                if isinstance(name, list) is False:
                    name = name.squeeze().tolist()
                if isinstance(name, int):
                    name_list.append(name)
                if isinstance(name, list):
                    for n in name:
                        name_list.append(n)
        
            if full_name:
                for k in full_name:
                    full_name_list.append(k)

            if tsne:
                for tsne_lb in tsne_label:
                    tsne_label_list.append(tsne_lb)
        except:
            pass

        with torch.no_grad():                
#             _, output = model_s1(images)
#             output = model(output)

            with gpytorch.settings.num_likelihood_samples(32):
                _, output = model_s1(images)
#                 import pdb;pdb.set_trace()
                output = model(output)
                output = output.to_data_independent_dist()
                output = likelihood(output).probs.mean(0)
#         import pdb;pdb.set_trace()
#         current_uncertainty = -(output * output.log()).sum(1)
        current_uncertainty = -(output * output.log()).sum(1) / torch.log(torch.tensor(output.shape[1], dtype=torch.float))
        current_uncertainty = current_uncertainty.cpu().numpy()
        
        if uncertainty is not None:
            uncertainty = np.concatenate((uncertainty, current_uncertainty))
        else:
            uncertainty = current_uncertainty
        
    return uncertainty, labels, image_paths, name_list, full_name_list, tsne_label_list

def set_model(args, args_due,train_com_loader , num_com):
    # stage 1
    input_size = 224
    num_classes = num_com
    
    if args_due.n_inducing_points is None:
        args_due.n_inducing_points = num_classes
    n_inducing_points = args_due.n_inducing_points
    
    if args_due.coeff == 1:
        from networks.mobilenetv3_SN1 import SupConMobileNetV3Large
    elif args_due.coeff == 3:
        from networks.mobilenetv3_SN3 import SupConMobileNetV3Large
    elif args_due.coeff == 5:
        from networks.mobilenetv3_SN5 import SupConMobileNetV3Large
    elif args_due.coeff == 7:
        from networks.mobilenetv3_SN7 import SupConMobileNetV3Large
    elif args_due.coeff == 0:
        from networks.mobilenetv3 import SupConMobileNetV3Large
    
    feature_extractor =  SupConMobileNetV3Large()

    initial_inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values(
            train_com_loader, feature_extractor, n_inducing_points
        )

    gp = dkl_Phison_mo.GP(
            num_outputs=num_classes, #可能=conponent 數量 = 23個 
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args_due.kernel,
    )

    gpmodel_s1 = dkl_Phison_mo.DKL(feature_extractor, gp)
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
    
    
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')


    if torch.cuda.is_available():
        gpmodel_s1 = gpmodel_s1.cuda()
        likelihood = likelihood.cuda()
        cudnn.benchmark = True
        gpmodel_s1.load_state_dict(ckpt)
    

    # stage 2 
    feature_extractor_s1 = gpmodel_s1.feature_extractor
    feature_extractor_s1.eval()
    
    initial_inducing_points, initial_lengthscale = dkl_Phison_mo_s2.initial_values3(
        train_com_loader, feature_extractor_s1, n_inducing_points*50 # if hparams.n_inducing_points= none ,hparams.n_inducing_points = num_class
    )
    
    print('initial_inducing_points : ', initial_inducing_points.shape)
    gp = dkl_Phison_mo_s2.GP(
        num_outputs=num_classes, #可能=conponent 數量 = 23個 
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=args_due.kernel,
    )

    gpmodel = dkl_Phison_mo_s2.DKL(gp)
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

    ckpt_gp = torch.load(args["gp_checkpoint_path"], map_location='cpu')
    
    if torch.cuda.is_available():
        gpmodel = gpmodel.cuda()
        likelihood = likelihood.cuda()
        cudnn.benchmark = True
        gpmodel.load_state_dict(ckpt_gp)

    
    return feature_extractor_s1, gpmodel, likelihood

def calculatePerformance(df, file_name):
    df['overkill_rate'] = (df['overkill'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage_rate'] = (df['leakage'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             'bad':sum(df['bad']),
             'overkill':sum(df['overkill']), 
             'leakage':sum(df['leakage']), 
             'overkill_rate':str(round(100*(sum(df['overkill'])/sum(df['total'])),5))+'%', 
             'leakage_rate': str(round(100*(sum(df['leakage'])/sum(df['total'])),5))+'%', 
            'unknown_rate': str(round(100*(sum(df['unknown'])/sum(df['total'])),5))+'%'}])], sort=False)
    df.to_csv(file_name, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-gp_c", "--gp_checkpoint_path",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--embedding_layer",
        type=str,
        default="shared_embedding",
        help="Which embedding to visualization( encoder or shared_embedding)"
    )
    parser.add_argument(
        "--name",
        type=int,
        default=15,
        help="Test component name"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed"
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
        "--no_calculate_uncertainty",
        action="store_false",
        dest="test_uncertainty",
        help="Don't use testing set on T-sne",
    )
    parser.add_argument(
        "--no_inference",
        action="store_false",
        dest="test_inference",
        help="Don't inference",
    )
    parser.add_argument(
        "--coeff", type=float, default=1, help="Spectral normalization coefficient"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--n_power_iterations", default=1, type=int, help="Number of power iterations"
    )
    parser.add_argument(
        "--kernel",
        default="RBF",
        choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
        help="Pick a kernel",
    )
    parser.add_argument(
        "--dataset",
        default="PHISON_regroup3",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','PHISON_regroup2','PHISON_regroup3'],
        help="Pick a dataset",
    )
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--dataset2', type=str, default='phison',
                        choices=['cifar10', 'cifar100', 'phison'], help='dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )
    parser.add_argument('--relabel', action='store_true', help='relabel dataset')
    parser.add_argument(
        "-oid", "--output_inference_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )

    
    args: Dict[str, Any] = vars(parser.parse_args())
    args_due = parser.parse_args()
    
    print('relabel:',args_due.relabel)
    set_random_seed(args["random_seed"])

    
    # Create output directory if not exists
    if not os.path.isdir(args["output_dir"]):
        os.makedirs(args["output_dir"])
        logging.info(f"Created output directory {args['output_dir']}")
        
    if not os.path.isdir('./output/'+args_due.output_inference_dir):
        os.makedirs('./output/'+args_due.output_inference_dir)

        

    # Initialize device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initialized device {device}")

    # Load model's checkpoint
    loc = 'cuda:0'
    
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")
    


    # Initialize dataset and dataloader
    df = pd.read_json(f"./bay/{args_due.output_inference_dir}_tau1_tau2_logs.json",lines=True)
    std_threshold_dict = df.iloc[df['target'].idxmax()].tolist()[1]

#     exp_2_tau1 = std_threshold_dict['exp_2_tau1'] 
#     exp_2_tau2 = std_threshold_dict['exp_2_tau2']
    com_0_tau1 = std_threshold_dict['com_0_tau1']
    com_0_tau2 = std_threshold_dict['com_0_tau2']
    com_1_tau1 = std_threshold_dict['com_1_tau1']
    com_1_tau2 = std_threshold_dict['com_1_tau2']
    com_2_tau1 = std_threshold_dict['com_2_tau1']
    com_2_tau2 = std_threshold_dict['com_2_tau2']
    com_3_tau1 = std_threshold_dict['com_3_tau1']
    com_3_tau2 = std_threshold_dict['com_3_tau2']
    com_4_tau1 = std_threshold_dict['com_4_tau1']
    com_4_tau2 = std_threshold_dict['com_4_tau2']
    com_5_tau1 = std_threshold_dict['com_5_tau1']
    com_5_tau2 = std_threshold_dict['com_5_tau2']
    com_6_tau1 = std_threshold_dict['com_6_tau1']
    com_6_tau2 = std_threshold_dict['com_6_tau2']
    com_7_tau1 = std_threshold_dict['com_7_tau1']
    com_7_tau2 = std_threshold_dict['com_7_tau2']
    com_8_tau1 = std_threshold_dict['com_8_tau1']
    com_8_tau2 = std_threshold_dict['com_8_tau2']
    com_9_tau1 = std_threshold_dict['com_9_tau1']
    com_9_tau2 = std_threshold_dict['com_9_tau2']
    com_10_tau1 = std_threshold_dict['com_10_tau1']
    com_10_tau2 = std_threshold_dict['com_10_tau2']
    com_11_tau1 = std_threshold_dict['com_11_tau1']
    com_11_tau2 = std_threshold_dict['com_11_tau2']
    com_12_tau1 = std_threshold_dict['com_12_tau1']
    com_12_tau2 = std_threshold_dict['com_12_tau2']
    com_13_tau1 = std_threshold_dict['com_13_tau1']
    com_13_tau2 = std_threshold_dict['com_13_tau2']
    com_14_tau1 = std_threshold_dict['com_14_tau1']
    com_14_tau2 = std_threshold_dict['com_14_tau2']

    df_train = pd.DataFrame()
    
    df_train = df_train.append({'com':0,'TH':com_0_tau1 , 'TH_2':com_0_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':1,'TH':com_1_tau1 , 'TH_2':com_1_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':2,'TH':com_2_tau1 , 'TH_2':com_2_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':3,'TH':com_3_tau1 , 'TH_2':com_3_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':4,'TH':com_4_tau1 , 'TH_2':com_4_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':5,'TH':com_5_tau1 , 'TH_2':com_5_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':6,'TH':com_6_tau1 , 'TH_2':com_6_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':7,'TH':com_7_tau1 , 'TH_2':com_7_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':8,'TH':com_8_tau1 , 'TH_2':com_8_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':9,'TH':com_9_tau1 , 'TH_2':com_9_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':10,'TH':com_10_tau1 , 'TH_2':com_10_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':11,'TH':com_11_tau1 , 'TH_2':com_11_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':12,'TH':com_12_tau1 , 'TH_2':com_12_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':13,'TH':com_13_tau1 , 'TH_2':com_13_tau2 }, ignore_index=True)
    df_train = df_train.append({'com':14,'TH':com_14_tau1 , 'TH_2':com_14_tau2 }, ignore_index=True)

            
    filepath = f'./output/{args_due.output_inference_dir}/uncertainty.csv'
    df_train.to_csv(filepath)

    
