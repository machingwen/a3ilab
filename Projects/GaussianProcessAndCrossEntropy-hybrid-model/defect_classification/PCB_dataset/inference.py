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
        if args_due.dataset == 'PHISON_regroup_six':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_seed1212(args_due.random_seed,add_test=True)

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
            num = 0
            for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
                if k == 0:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==1:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'missing'] = temp_data['class'].value_counts().sort_index().values[num]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'missing'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==2:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'shift'] = temp_data['class'].value_counts().sort_index().values[num]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'shift'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==3:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'stand'] = temp_data['class'].value_counts().sort_index().values[num]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'stand'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==4:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'broke'] = temp_data['class'].value_counts().sort_index().values[num]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'broke'] = temp_data['class'].value_counts().sort_index().values[0]
                elif k ==5:
                    try:
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'short'] = temp_data['class'].value_counts().sort_index().values[num]
                    except:
                        print(f"{name} only contains bad label.")
                        test_component_name_df.loc[test_component_name_df['component_name'] == name, 'short'] = temp_data['class'].value_counts().sort_index().values[0]
                num = num + 1
        test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
        try:
            test_component_name_df['missing'] = test_component_name_df['missing'].fillna(0).astype(int)
            test_component_name_df['shift'] = test_component_name_df['shift'].fillna(0).astype(int)
            test_component_name_df['stand'] = test_component_name_df['stand'].fillna(0).astype(int)
            test_component_name_df['broke'] = test_component_name_df['broke'].fillna(0).astype(int)
            test_component_name_df['short'] = test_component_name_df['short'].fillna(0).astype(int)
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'missing', 'shift', 'stand', 'broke', 'short']]    
        except:
            test_component_name_df = test_component_name_df[['component_name', 'total', 'good']]       
        
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

def set_model1(args, args_due,train_com_loader,num_com):
#     model = SupConMobileNetV3Large(name='mobilenetv3_large')
    input_size = 224
    num_classes = num_com
    
    # if args_due.n_inducing_points is None:
    #     args_due.n_inducing_points = num_classes
    # n_inducing_points = args_due.n_inducing_points
    
    from networks.mobilenetv3_six import SupConMobileNetV3Large as SupConMobileNetV3Large1
    
    model =  SupConMobileNetV3Large1()

    # initial_inducing_points, initial_lengthscale = dkl_Phison_exp1.initial_values(
    #         train_com_loader, feature_extractor, n_inducing_points 
    #     )

    # gp = dkl_Phison_exp1.GP(
    #         num_outputs=num_com, #可能=conponent 數量 = 23個 
    #         initial_lengthscale=initial_lengthscale,
    #         initial_inducing_points=initial_inducing_points,
    #         kernel=args_due.kernel,
    # )

    # model = dkl_Phison_exp1.DKL(feature_extractor, gp)
    # likelihood = SoftmaxLikelihood(num_classes=num_com, mixing_weights=False)
    
    
    ckpt = torch.load(args["checkpoint_path1"], map_location='cpu')

#     for k, v in ckpt.items():  # k 参数名 v 对应参数值
#         print(k)

#     import pdb;pdb.set_trace()

    if torch.cuda.is_available():
        model = model.cuda()
        # likelihood = likelihood.cuda()
        cudnn.benchmark = True
        model.load_state_dict(ckpt)
        
    return model


def set_model2(args, args_due,train_com_loader , num_com):
    # stage 1
    input_size = 224
    num_classes = num_com
    
    if args_due.n_inducing_points is None:
        args_due.n_inducing_points = num_classes
    n_inducing_points = args_due.n_inducing_points
    
    
    if args_due.coeff == 1:
        from networks.mobilenetv3_SN1 import SupConMobileNetV3Large as SupConMobileNetV3Large2
    elif args_due.coeff == 3:
        from networks.mobilenetv3_SN3 import SupConMobileNetV3Large as SupConMobileNetV3Large2
    elif args_due.coeff == 5:
        from networks.mobilenetv3_SN5 import SupConMobileNetV3Large as SupConMobileNetV3Large2
    elif args_due.coeff == 7:
        from networks.mobilenetv3_SN7 import SupConMobileNetV3Large as SupConMobileNetV3Large2
    elif args_due.coeff == 0:
        from networks.mobilenetv3 import SupConMobileNetV3Large as SupConMobileNetV3Large2
    
    feature_extractor =  SupConMobileNetV3Large2()

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
    
    
    ckpt = torch.load(args["checkpoint_path2"], map_location='cpu')


    if torch.cuda.is_available():
        gpmodel_s1 = gpmodel_s1.cuda()
        likelihood = likelihood.cuda()
#         cudnn.benchmark = True
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
#         cudnn.benchmark = True
        gpmodel.load_state_dict(ckpt_gp)

    
    return feature_extractor_s1, gpmodel, likelihood

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
             'missing':sum(df['missing']),
             'shift':sum(df['shift']),
             'stand':sum(df['stand']),
             'broke':sum(df['broke']),
             'short':sum(df['short']),
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
             'unknown':sum(df['unknown']),
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
             'missing':sum(df['missing']),
             'shift':sum(df['shift']),
             'stand':sum(df['stand']),
             'broke':sum(df['broke']),
             'short':sum(df['short']),
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
             'unknown':sum(df['unknown']),
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

def EXP_old(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
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
            
            cls_output = model(images)
            _, prediction=torch.max(cls_output.data, 1)
            acc1 = accuracy(cls_output, labels, topk=(1))
            top1.update(acc1[0].item(), bsz)
            y_pred.extend(prediction.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            # ALL test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
#                 if gt.item() == 0 and gt.item() != pred.item():
#                     if isinstance(name, str):
#                         df_all.loc[(df_all["component_name"] == name), 'overkill'] +=1
#                     else:
#                         df_all.loc[(df_all["component_name"] == name.item()), 'overkill'] +=1
#                 if gt.item() !=0  and gt.item() != pred.item():
#                     if isinstance(name, str):
#                         df_all.loc[(df_all["component_name"] == name), 'leakage'] +=1
#                     else:
#                         df_all.loc[(df_all["component_name"] == name.item()), 'leakage'] +=1
                if gt.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill0'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill0'] +=1
                if pred.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage0'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage0'] +=1
                if gt.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill1'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill1'] +=1
                if pred.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage1'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage1'] +=1
                if gt.item() == 2 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill2'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill2'] +=1
                if pred.item() == 2 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage2'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage2'] +=1
                if gt.item() == 3 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill3'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill3'] +=1
                if pred.item() == 3 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage3'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage3'] +=1
                if gt.item() == 4 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill4'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill4'] +=1
                if pred.item() == 4 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage4'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage4'] +=1
                if gt.item() == 5 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill5'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill5'] +=1
                if pred.item() == 5 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage5'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage5'] +=1


    print(' * EXP1_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_old_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred
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
            
            cls_output, component_out = model(images)
            _, prediction=torch.max(cls_output.data, 1)
            acc1 = accuracy(cls_output, labels, topk=(1))
            top1.update(acc1[0].item(), bsz)
            y_pred.extend(prediction.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            # ALL test samples
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                if gt.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill0'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill0'] +=1
                if pred.item() == 0 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage0'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage0'] +=1
                if gt.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill1'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill1'] +=1
                if pred.item() == 1 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage1'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage1'] +=1
                if gt.item() == 2 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill2'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill2'] +=1
                if pred.item() == 2 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage2'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage2'] +=1
                if gt.item() == 3 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill3'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill3'] +=1
                if pred.item() == 3 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage3'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage3'] +=1
                if gt.item() == 4 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill4'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill4'] +=1
                if pred.item() == 4 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage4'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage4'] +=1
                if gt.item() == 5 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'overkill5'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'overkill5'] +=1
                if pred.item() == 5 and gt.item() != pred.item():
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'leakage5'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'leakage5'] +=1
                        
    print(' * EXP1_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP1_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred

def EXP2(args_due , model_s1,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ,likelihood ,TH , TH_2,good_num_com):
    
    model.eval()
    model_s1.eval()

    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    com_pred = []
    y_true = []
    y_com = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, labels, _, component_name) in enumerate(tqdm(test_loader)):
            images = images.float().cuda()
            labels = labels.cuda()
            component_name = component_name.cuda()
            bsz = labels.shape[0]
            
            
            with gpytorch.settings.num_likelihood_samples(1000):
                _, component_out = model_s1(images)
                component_out = model(component_out)
                component_out = component_out.to_data_independent_dist()
                component_out = likelihood(component_out).probs.mean(0)

#             current_uncertainty = -(component_out * component_out.log()).sum(1)
            current_uncertainty = -(component_out * component_out.log()).sum(1) / torch.log(torch.tensor(component_out.shape[1], dtype=torch.float))
            
            
            _, prediction=torch.max(component_out.data, 1)
#             prediction = [21 if x >= 21 else x for x in prediction]
#             component_name = [21 if x >= 21 else x for x in component_name]
            com_pred.extend(prediction.view(-1).detach().cpu().numpy())
            
            unk_list=[]
            bad_list=[]
            for i in range(len(current_uncertainty)):
                uncertainty_th = TH[prediction[i].item()]
                uncertainty_th_2 = TH_2[prediction[i].item()]

#                 if current_uncertainty[i]>=uncertainty_th :
#                     bad_list.append(i)
                if current_uncertainty[i]>=uncertainty_th and current_uncertainty[i]<=uncertainty_th_2:
                    bad_list.append(i)
                if current_uncertainty[i]>uncertainty_th_2:
                    unk_list.append(i)
#                 if current_uncertainty[i]>=uncertainty_th :
#                     bad_list.append(i)
                
                    

                
            prediction = [0 if x < good_num_com else 1 for x in prediction] 
            for i in bad_list:
                prediction[i] = 1    
            for i in unk_list:
                prediction[i] = 2   

#             acc1 = accuracy(component_out, component_name, topk=(1))
#             top1.update(acc1[0].item(), bsz)
#             import pdb;pdb.set_trace()
#             y_pred.extend(prediction.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            y_com.extend(component_name)
#             y_com.extend(component_name.view(-1).detach().cpu().numpy())
            

            
            prediction=torch.Tensor(prediction)
#             import pdb;pdb.set_trace()
            y_pred.extend(prediction.view(-1).detach().cpu().numpy())
             
            # ALL test samples 
            for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                
                if pred.item() == 2:
                    if isinstance(name, str):
                        df_all.loc[(df_all["component_name"] == name), 'unknown'] +=1
                    else:
                        df_all.loc[(df_all["component_name"] == name.item()), 'unknown'] +=1
                if pred.item() != 2:
                    if gt.item() == 0 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_all.loc[(df_all["component_name"] == name), 'overkill0'] +=1
                        else:
                            df_all.loc[(df_all["component_name"] == name.item()), 'overkill0'] +=1
                    if gt.item() == 1 and gt.item() != pred.item():
                        if isinstance(name, str):
                            df_all.loc[(df_all["component_name"] == name), 'leakage0'] +=1
                        else:
                            df_all.loc[(df_all["component_name"] == name.item()), 'leakage0'] +=1


#     print(' * EXP2_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance_unk(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP2_ALL_test_set_overkill_and_leakage_unk.csv')
    
    return y_com, y_pred, com_pred

def HybridExpert(y_pred_expert_old, y_pred_expert1, y_pred_expert2, com_pred_expert2, y_true, name_label_list,  df, args_due, train_component_label, val_component_label, test_component_label):

    
    gt_label_after_unknown = []
    pred_after_unknown = []
    
    y_pred_all = []
    y_true_all = []
    
    prediction_IND = []
    gt_label_IND = []
    prediction_OOD = []
    gt_label_OOD = []
    
    df_ind = df.copy()
    df_ood = df.copy()
    
#     if args_due.random_seed == 2 : 
#         new_component = [2,6,7]
#     if args_due.random_seed == 211 :
#         new_component = [1,7,8]
#     if args_due.random_seed == 211 :
#         new_component = [3,7,10]
    
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]
    
            
    for idx, (pred_expert_old, pred_expert1, pred_expert2,com_pred, gt_label, name) in enumerate(list(zip(y_pred_expert_old, y_pred_expert1, y_pred_expert2, com_pred_expert2, y_true, name_label_list))):
        name = np.int64(name.item())
        if name not in test_component_label :
            
            y_pred_all.append(pred_expert_old)
            y_true_all.append(gt_label)
            
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(pred_expert_old)
            
            if gt_label == 0 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill0'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill0'] +=1
            elif pred_expert_old == 0 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage0'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage0'] +=1 
            if gt_label == 1 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill1'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill1'] +=1
            elif pred_expert_old == 1 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage1'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage1'] +=1 
            if gt_label == 2 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill2'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill2'] +=1
            elif pred_expert_old == 2 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage2'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage2'] +=1 
            if gt_label == 3 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill3'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill3'] +=1
            elif pred_expert_old == 3 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage3'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage3'] +=1 
            if gt_label == 4 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill4'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill4'] +=1
            elif pred_expert_old == 4 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage4'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage4'] +=1 
            if gt_label == 5 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'overkill5'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'overkill5'] +=1
            elif pred_expert_old == 5 and gt_label != pred_expert_old:
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'leakage5'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'leakage5'] +=1
                    
        else :
            if ((pred_expert1==0) and (pred_expert2!=0)) or ((pred_expert1 != 0) and (pred_expert2==0)) or (pred_expert2 ==2):
#             import pdb;pdb.set_trace()
                y_pred_all.append(6)
                y_true_all.append(gt_label)
            
                if isinstance(name, np.int64):
                    df.loc[(df["component_name"] == name), 'unknown'] +=1
                else:
                    df.loc[(df["component_name"] == ''.join(name)), 'unknown'] +=1

            if (pred_expert1 == pred_expert2) or ((pred_expert1 != 0) and (pred_expert2 ==1)):

                y_pred_all.append(pred_expert1)
                y_true_all.append(gt_label)
            
                gt_label_after_unknown.append(gt_label)
                pred_after_unknown.append(pred_expert1)

                if gt_label == 0 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'overkill0'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'overkill0'] +=1
                elif pred_expert1 == 0 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'leakage0'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'leakage0'] +=1 
                if gt_label == 1 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'overkill1'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'overkill1'] +=1
                elif pred_expert1 == 1 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'leakage1'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'leakage1'] +=1 
                if gt_label == 2 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'overkill2'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'overkill2'] +=1
                elif pred_expert1 == 2 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'leakage2'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'leakage2'] +=1 
                if gt_label == 3 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'overkill3'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'overkill3'] +=1
                elif pred_expert1 == 3 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'leakage3'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'leakage3'] +=1 
                if gt_label == 4 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'overkill4'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'overkill4'] +=1
                elif pred_expert1 == 4 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'leakage4'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'leakage4'] +=1 
                if gt_label == 5 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'overkill5'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'overkill5'] +=1
                elif pred_expert1 == 5 and gt_label != pred_expert1:
                    if isinstance(name, np.int64):
                        df.loc[(df["component_name"] == name), 'leakage5'] +=1
                    else:
                        df.loc[(df["component_name"] == ''.join(name)), 'leakage5'] +=1 
            
#         if ((pred_expert1==0) and (pred_expert2!=0)) or ((pred_expert1 != 0) and (pred_expert2==0)) or (pred_expert2 ==2):
# #             import pdb;pdb.set_trace()
#             y_pred_all.append(6)
#             y_true_all.append(gt_label)
            
#             if isinstance(name, np.int64):
#                 df.loc[(df["component_name"] == name), 'unknown'] +=1
#             else:
#                 df.loc[(df["component_name"] == ''.join(name)), 'unknown'] +=1

#         if (pred_expert1 == pred_expert2) or ((pred_expert1 != 0) and (pred_expert2 ==1)):

#             y_pred_all.append(pred_expert1)
#             y_true_all.append(gt_label)
            
#             gt_label_after_unknown.append(gt_label)
#             pred_after_unknown.append(pred_expert1)

#             if gt_label == 0 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'overkill0'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'overkill0'] +=1
#             elif pred_expert1 == 0 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'leakage0'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'leakage0'] +=1 
#             if gt_label == 1 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'overkill1'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'overkill1'] +=1
#             elif pred_expert1 == 1 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'leakage1'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'leakage1'] +=1 
#             if gt_label == 2 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'overkill2'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'overkill2'] +=1
#             elif pred_expert1 == 2 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'leakage2'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'leakage2'] +=1 
#             if gt_label == 3 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'overkill3'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'overkill3'] +=1
#             elif pred_expert1 == 3 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'leakage3'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'leakage3'] +=1 
#             if gt_label == 4 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'overkill4'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'overkill4'] +=1
#             elif pred_expert1 == 4 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'leakage4'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'leakage4'] +=1 
#             if gt_label == 5 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'overkill5'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'overkill5'] +=1
#             elif pred_expert1 == 5 and gt_label != pred_expert1:
#                 if isinstance(name, np.int64):
#                     df.loc[(df["component_name"] == name), 'leakage5'] +=1
#                 else:
#                     df.loc[(df["component_name"] == ''.join(name)), 'leakage5'] +=1 

    print("HybridExpert Accuracy: {}\n".format(100*round(accuracy_score(gt_label_after_unknown, pred_after_unknown),4)))  
    calculatePerformance_unk(df, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_HBE_ALL_test_set_overkill_and_leakage_unk.csv')


if __name__ == "__main__":


    start = time.time()

    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-c1", "--checkpoint_path1",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-c2", "--checkpoint_path2",
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
    
    checkpoint_path1: str = args["checkpoint_path1"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path1, map_location="cuda:0")
#     checkpoint: Dict[str, Any] = torch.load(checkpoint_path1, map_location="cpu")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path1']}")
    
    checkpoint_path2: str = args["checkpoint_path2"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path2, map_location="cuda:0")
#     checkpoint: Dict[str, Any] = torch.load(checkpoint_path2, map_location="cpu")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path2']}")
    
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
    model1  = set_model1(args, args_due, train_com_loader , num_classes)
    model, gp_model, likelihood  = set_model2(args, args_due, train_com_loader , num_classes)
    
    # set TH 
    df = pd.read_csv(f'./output/{args_due.output_inference_dir}/uncertainty.csv')
    TH = df['TH']
    TH_2 = df['TH_2']
    
    # Initialize dataset and dataloader
    
    if args_due.test_inference == True:
               
        
        val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label = set_loader(args_due)

        test_df_orig = test_component_name_df.copy()    
        test_component_name_df_Expert2 = test_df_orig.copy()
        test_component_name_df_HybridExpert = test_df_orig.copy()
        
        print("1. Testing with Discriminative Model (Expert 1)")  # Discriminative model
        y_true, y_pred_expert_old = EXP_old(args_due ,model1 ,test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)

        print("1. Testing with Discriminative Model (Expert 1)")  # Discriminative model
        y_true, y_pred_expert1 = EXP1(args_due ,model ,test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)

        print("2. Testing with DUE (Expert 2)")  # Discriminative model
        name_label_list, y_pred_expert2, com_pred_expert2  = EXP2(args_due , model,gp_model ,test_loader, test_component_name_df_Expert2, train_component_label, val_component_label, test_component_label ,likelihood ,TH,TH_2 ,good_num_com)

        print("3. Testing with Hybrid Expert") 
        HybridExpert(y_pred_expert_old, y_pred_expert1, y_pred_expert2, com_pred_expert2, y_true, name_label_list, test_component_name_df_HybridExpert, args_due, train_component_label, val_component_label, test_component_label)
    else:
        print('inference calculations are not performed')
        

    end = time.time()

    print("執行時間：%f 秒" % (end - start))

    
