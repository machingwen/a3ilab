# -*- coding: utf-8 -*-
# https://github.com/QuocThangNguyen/deep-metric-learning-tsinghua-dogs/blob/master/src/scripts/visualize_tsne.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
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

from tqdm import tqdm

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
#     datefmt='%y-%b-%d %H:%M:%S',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )

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

def set_loader(args_due):
    # construct data loader
    if args_due.dataset2 == 'phison':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(args_due.dataset2))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize([args_due.size, args_due.size]),
        transforms.ToTensor(),
        normalize,
    ])
    if args_due.dataset2 == 'phison':
        if args_due.dataset == 'PHISON_regroup':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due(args_due.random_seed,add_test=True) 
        if args_due.dataset == 'PHISON_regroup2':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2(args_due.random_seed,add_test=True, testing=None) 
        if args_due.dataset == 'PHISON':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_relabel(args_due.random_seed, testing=None)  
        if args_due.dataset == 'PHISON_regroup3':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_seed1212(args_due.random_seed,add_test=True)
        if args_due.dataset == 'PHISON_regroup_six':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_sixcls(args_due.random_seed,add_test=True)

        train_dataset = CustomDataset(train_df, transform=val_transform)
        val_dataset = CustomDataset(val_df, transform=val_transform)
        test_dataset = CustomDataset(test_df, transform=val_transform)
        
        train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

#         test_df = pd.concat([train_df, val_df])

        test_df_mapping2_label = test_df.copy()    
        name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
        num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
        test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

        for name in set(test_df_mapping2_label['component_name'].values):
            temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
            for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
                if k == 0:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==1:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'missing'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains missing label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'missing'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==2:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'shift'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains shift label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'shift'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==3:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'stand'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains stand label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'stand'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==4:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'short'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains short label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'short'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==5:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
        test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
        try:
            test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
        except:
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good']]        
        
#         col = {'overkill': 0, 'leakage': 0, 'unknown':  0,}
        col = {'overkill0': 0, 'leakage0': 0,
               'overkill1': 0, 'leakage1': 0,
               'overkill2': 0, 'leakage2': 0,
               'overkill3': 0, 'leakage3': 0,
               'overkill4': 0, 'leakage4': 0,
               'overkill5': 0, 'leakage5': 0,
               'unknown':  0,}
        test_component_name_df = test_component_name_df.assign(**col)
        
    else:
        raise ValueError(args_due.dataset2)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args_due.batch_size, shuffle=False,
        num_workers=args_due.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args_due.batch_size, shuffle=False,
        num_workers=args_due.num_workers, pin_memory=True)
    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=args_due.batch_size, shuffle=False,
        num_workers=args_due.num_workers, pin_memory=True)
    
    return val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label 

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
    
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
    
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')
    
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
    
    if torch.cuda.is_available():
        feature_extractor.cuda()
        #gpmodel_s1 = gpmodel_s1.cuda()
        likelihood = likelihood.cuda()
#         cudnn.benchmark = True
        #gpmodel_s1.load_state_dict(ckpt)

    
    return feature_extractor, likelihood

#

def calculatePerformance(df, file_name):
    df['overkill0_rate'] = (df['overkill0'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage0_rate'] = (df['leakage0'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill1_rate'] = (df['overkill1'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage1_rate'] = (df['leakage1'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill2_rate'] = (df['overkill2'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage2_rate'] = (df['leakage2'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill3_rate'] = (df['overkill3'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage3_rate'] = (df['leakage3'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill4_rate'] = (df['overkill4'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage4_rate'] = (df['leakage4'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill5_rate'] = (df['overkill5'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage5_rate'] = (df['leakage5'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             'bad':sum(df['bad']),
             'overkill0':sum(df['overkill0']), 
             'leakage0':sum(df['leakage0']),
             'overkill1':sum(df['overkill1']), 
             'leakage1':sum(df['leakage1']),
             'overkill2':sum(df['overkill2']), 
             'leakage2':sum(df['leakage2']),
             'overkill3':sum(df['overkill3']), 
             'leakage3':sum(df['leakage3']),
             'overkill4':sum(df['overkill4']), 
             'leakage4':sum(df['leakage4']),
             'overkill5':sum(df['overkill5']), 
             'leakage5':sum(df['leakage5']),
             'overkill0_rate':str(round(100*(sum(df['overkill0'])/sum(df['total'])),5))+'%', 
             'leakage0_rate': str(round(100*(sum(df['leakage0'])/sum(df['total'])),5))+'%',
             'overkill1_rate':str(round(100*(sum(df['overkill1'])/sum(df['total'])),5))+'%', 
             'leakage1_rate': str(round(100*(sum(df['leakage1'])/sum(df['total'])),5))+'%',
             'overkill2_rate':str(round(100*(sum(df['overkill2'])/sum(df['total'])),5))+'%', 
             'leakage2_rate': str(round(100*(sum(df['leakage2'])/sum(df['total'])),5))+'%',
             'overkill3_rate':str(round(100*(sum(df['overkill3'])/sum(df['total'])),5))+'%', 
             'leakage3_rate': str(round(100*(sum(df['leakage3'])/sum(df['total'])),5))+'%',
             'overkill4_rate':str(round(100*(sum(df['overkill4'])/sum(df['total'])),5))+'%', 
             'leakage4_rate': str(round(100*(sum(df['leakage4'])/sum(df['total'])),5))+'%',
             'overkill5_rate':str(round(100*(sum(df['overkill5'])/sum(df['total'])),5))+'%', 
             'leakage5_rate': str(round(100*(sum(df['leakage5'])/sum(df['total'])),5))+'%',
             'unknown_rate': str(round(100*(sum(df['unknown'])/sum(df['total'])),5))+'%'}])], sort=False)
    df.to_csv(file_name, index=False)

def calculatePerformance_unk(df, file_name):
#     df['overkill_rate'] = (df['overkill'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
#     df['leakage_rate'] = (df['leakage'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill0_rate'] = (df['overkill0'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage0_rate'] = (df['leakage0'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill1_rate'] = (df['overkill1'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage1_rate'] = (df['leakage1'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill2_rate'] = (df['overkill2'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage2_rate'] = (df['leakage2'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill3_rate'] = (df['overkill3'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage3_rate'] = (df['leakage3'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill4_rate'] = (df['overkill4'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage4_rate'] = (df['leakage4'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['overkill5_rate'] = (df['overkill5'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df['leakage5_rate'] = (df['leakage5'] / df['total'] * 100).round(decimals = 5).astype(str) + '%'
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             'bad':sum(df['bad']),
             'overkill0':sum(df['overkill0']), 
             'leakage0':sum(df['leakage0']),
             'overkill1':sum(df['overkill1']), 
             'leakage1':sum(df['leakage1']),
             'overkill2':sum(df['overkill2']), 
             'leakage2':sum(df['leakage2']),
             'overkill3':sum(df['overkill3']), 
             'leakage3':sum(df['leakage3']),
             'overkill4':sum(df['overkill4']), 
             'leakage4':sum(df['leakage4']),
             'overkill5':sum(df['overkill5']), 
             'leakage5':sum(df['leakage5']),
             'overkill0_rate':str(round(100*(sum(df['overkill0'])/sum(df['total'])),5))+'%', 
             'leakage0_rate': str(round(100*(sum(df['leakage0'])/sum(df['total'])),5))+'%',
             'overkill1_rate':str(round(100*(sum(df['overkill1'])/sum(df['total'])),5))+'%', 
             'leakage1_rate': str(round(100*(sum(df['leakage1'])/sum(df['total'])),5))+'%',
             'overkill2_rate':str(round(100*(sum(df['overkill2'])/sum(df['total'])),5))+'%', 
             'leakage2_rate': str(round(100*(sum(df['leakage2'])/sum(df['total'])),5))+'%',
             'overkill3_rate':str(round(100*(sum(df['overkill3'])/sum(df['total'])),5))+'%', 
             'leakage3_rate': str(round(100*(sum(df['leakage3'])/sum(df['total'])),5))+'%',
             'overkill4_rate':str(round(100*(sum(df['overkill4'])/sum(df['total'])),5))+'%', 
             'leakage4_rate': str(round(100*(sum(df['leakage4'])/sum(df['total'])),5))+'%',
             'overkill5_rate':str(round(100*(sum(df['overkill5'])/sum(df['total'])),5))+'%', 
             'leakage5_rate': str(round(100*(sum(df['leakage5'])/sum(df['total'])),5))+'%',
#              'c_overkill_rate':str(round(100*(sum(df['overkill'])/(sum(df['total'])-sum(df['unknown']))),5))+'%', 
#              'c_leakage_rate': str(round(100*(sum(df['leakage'])/(sum(df['total'])-sum(df['unknown']))),5))+'%', 
             'unknown_rate': str(round(100*(sum(df['unknown'])/sum(df['total'])),5))+'%'}])], sort=False)
    df.to_csv(file_name, index=False)

def EXP1(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
    model.eval()
    
    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    y_true = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, labels, _, component_name) in enumerate(tqdm(test_loader)):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            
#             import pdb;pdb.set_trace()
            cls_output = model(images)
            _, prediction=torch.max(cls_output.data, 1)
            acc1 = accuracy(cls_output, labels, topk=(1))
            top1.update(acc1[0].item(), bsz)
            y_pred.extend(prediction.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            # ALL test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                if gt.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill'] +=1
                if gt.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage'] +=1
                        
    print(' * EXP1_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP1_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred



if __name__ == "__main__":


    start = time.time()

    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-c", "--checkpoint_path",
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
        default="PHISON_regroup_six",
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
#     checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")
    
    # load data
    ds = get_dataset(args_due.dataset ,args_due.random_seed , root="./data" )
#     input_size, num_classes, train_dataset, test_dataset, train_loader, train_com_loader = ds
    input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset = ds
    
    # Intialize model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/clust/1212_pretrain_all_clustimage_model')
    good_num_com = len(set(clust.results['labels']))
    num_com = len(set(clust.results['labels']))+2
    print(num_classes)
    model, likelihood  = set_model(args, args_due, train_com_loader , num_classes)
    
    
    # Initialize dataset and dataloader
    
    if args_due.test_inference == True:
               
        val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label = set_loader(args_due)

        test_df_orig = test_component_name_df.copy()    
        test_component_name_df_Expert2 = test_df_orig.copy()
        test_component_name_df_HybridExpert = test_df_orig.copy()

        print("1. Testing with Discriminative Model (Expert 1)")  # Discriminative model
        y_true, y_pred_expert1 = EXP1(args_due ,model ,test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)

    else:
        print('inference calculations are not performed')
        

    end = time.time()

    print("執行時間：%f 秒" % (end - start))

    
