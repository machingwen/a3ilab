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
from sklearn.metrics import confusion_matrix

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
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_relabel_sixcls_randomsplit(args_due.random_seed, testing=None) 
        if args_due.dataset == 'PHISON_fourcls':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, _ = CreateDataset_relabel_fourcls(args_due.random_seed,add_test=True, testing=None)
        if args_due.dataset == 'PHISON_shift':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, _ = CreateDataset_relabel_shift_randomsplit(args_due.random_seed, testing=None) 
        if args_due.dataset == 'PHISON_broke':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, _ = CreateDataset_relabel_broke_randomsplit(args_due.random_seed, testing=None) 
        if args_due.dataset == 'PHISON_short':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, _ = CreateDataset_relabel_short(args_due.random_seed,add_test=True, testing=None)
        if args_due.dataset == 'PHISON_regroup3':
            train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_seed1212(args_due.random_seed,add_test=True)
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
        
        col = {'TP0': 0, 'TN0': 0, 'FP0': 0, 'FN0': 0,
               'TP1': 0, 'TN1': 0, 'FP1': 0, 'FN1': 0,
               'TP2': 0, 'TN2': 0, 'FP2': 0, 'FN2': 0,
               'TP3': 0, 'TN3': 0, 'FP3': 0, 'FN3': 0,
               'TP4': 0, 'TN4': 0, 'FP4': 0, 'FN4': 0,
               'TP5': 0, 'TN5': 0, 'FP5': 0, 'FN5': 0,
#                'overkill1': 0, 'leakage1': 0,
#                'overkill2': 0, 'leakage2': 0,
#                'overkill3': 0, 'leakage3': 0,
#                'overkill4': 0, 'leakage4': 0,
#                'overkill5': 0, 'leakage5': 0,
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
    
    feature_extractor_s1 =  SupConMobileNetV3Large()
    feature_extractor_s2 =  SupConMobileNetV3Large()

    initial_inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values(
            train_com_loader, feature_extractor_s1, n_inducing_points
        )

    gp = dkl_Phison_mo.GP(
            num_outputs=num_classes, #可能=conponent 數量 = 23個 
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args_due.kernel,
    )

    gpmodel_s1 = dkl_Phison_mo.DKL(feature_extractor_s1, gp)
    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
    
#     feature_extractor_s2 = feature_extractor
    
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')
    ckpt_mlp = torch.load(args["mlp_checkpoint_path"], map_location='cpu')


    if torch.cuda.is_available():
        gpmodel_s1 = gpmodel_s1.cuda()
        feature_extractor_s2 = feature_extractor_s2.cuda()
        likelihood = likelihood.cuda()
#         cudnn.benchmark = True
        gpmodel_s1.load_state_dict(ckpt)
        feature_extractor_s2.load_state_dict(ckpt_mlp)
    

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

    
    return feature_extractor_s1,feature_extractor_s2, gpmodel, likelihood

#

def calculatePerformance(df, file_name):

    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             # 'shift':sum(df['shift']),
             # 'broke':sum(df['broke']),
             # 'short':sum(df['short']),
             'TP0':sum(df['TP0']),'TN0':sum(df['TN0']),'FP0':sum(df['FP0']),'FN0':sum(df['FN0']),
             'TP1':sum(df['TP1']),'TN1':sum(df['TN1']),'FP1':sum(df['FP1']),'FN1':sum(df['FN1']),
             'TP2':sum(df['TP2']),'TN2':sum(df['TN2']),'FP2':sum(df['FP2']),'FN2':sum(df['FN2']),
             'TP3':sum(df['TP3']),'TN3':sum(df['TN3']),'FP3':sum(df['FP3']),'FN3':sum(df['FN3']),
             'TP4':sum(df['TP4']),'TN4':sum(df['TN4']),'FP4':sum(df['FP4']),'FN4':sum(df['FN4']),
             'TP5':sum(df['TP5']),'TN5':sum(df['TN5']),'FP5':sum(df['FP5']),'FN5':sum(df['FN5']),
             'unknown':sum(df['unknown']),             
            'unknown_rate': str(round(100*(sum(df['unknown'])/sum(df['total'])),5))+'%'}])], sort=False)
    df.to_csv(file_name, index=False)

def calculatePerformance_unk(df, file_name):
#     
    df = pd.concat([df, pd.DataFrame.from_records([
            {'total':sum(df['total']),
             'good':sum(df['good']),
             
             'TP0':sum(df['TP0']),'TN0':sum(df['TN0']),'FP0':sum(df['FP0']),'FN0':sum(df['FN0']),
             'TP1':sum(df['TP1']),'TN1':sum(df['TN1']),'FP1':sum(df['FP1']),'FN1':sum(df['FN1']),
             'TP2':sum(df['TP2']),'TN2':sum(df['TN2']),'FP2':sum(df['FP2']),'FN2':sum(df['FN2']),
             'TP3':sum(df['TP3']),'TN3':sum(df['TN3']),'FP3':sum(df['FP3']),'FN3':sum(df['FN3']),
             'TP4':sum(df['TP4']),'TN4':sum(df['TN4']),'FP4':sum(df['FP4']),'FN4':sum(df['FN4']),
             'TP5':sum(df['TP5']),'TN5':sum(df['TN5']),'FP5':sum(df['FP5']),'FN5':sum(df['FN5']),
#             
            'unknown_rate': str(round(100*(sum(df['unknown'])/sum(df['total'])),5))+'%'}])], sort=False)
    df.to_csv(file_name, index=False)

def EXP_detector_shift(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
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
        for idx, (images, _, labels, _, _, _, component_name) in enumerate(tqdm(test_loader)):
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
            # for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                     
                        
    print(' * EXP_detector_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_detector_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred

def EXP_detector_broke(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
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
        for idx, (images, _, _, labels, _, _, component_name) in enumerate(tqdm(test_loader)):
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
            # for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                     
                        
    print(' * EXP_detector_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_detector_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred

def EXP_detector_short(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
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
        for idx, (images, _, _, _, labels, _, component_name) in enumerate(tqdm(test_loader)):
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
            # for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                     
                        
    print(' * EXP_detector_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_detector_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred

def EXP_classifier(args_due ,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ):
    
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
        for idx, (images, _, labels, _, component_name) in enumerate(tqdm(test_loader)):
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
            # for idx, (gt, pred, name) in enumerate(list(zip(labels.data, prediction.data, component_name))):
                
                
    print(' * EXP_classifier_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_classifier_ALL_test_set_overkill_and_leakage.csv')
    
    return y_true, y_pred

def EXP_component_shift(args_due , model_s1,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ,likelihood ,TH , TH_2):
    
    model.eval()
    model_s1.eval()

    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    y_true = []
    y_com = []
    com_pred = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, _, labels, _, _, _, component_name) in enumerate(tqdm(test_loader)):
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
            com_pred.extend(prediction.view(-1).detach().cpu().numpy())
            
            
            unk_list=[]
            bad_list=[]
            for i in range(len(current_uncertainty)):
                uncertainty_th = TH[prediction[i].item()]
                uncertainty_th_2 = TH_2[prediction[i].item()]

                if current_uncertainty[i]>=uncertainty_th and current_uncertainty[i]<=uncertainty_th_2:
                    bad_list.append(i)
                if current_uncertainty[i]>uncertainty_th_2:
                    unk_list.append(i)
                
                    

                
            prediction = [0 for x in prediction] 
            for i in bad_list:
                prediction[i] = 1    
            for i in unk_list:
                prediction[i] = 2   

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


#     print(' * EXP2_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance_unk(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_component_ALL_test_set_overkill_and_leakage_unk.csv')
    
    return y_com, y_pred, com_pred

def EXP_component_broke(args_due , model_s1,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ,likelihood ,TH , TH_2):
    
    model.eval()
    model_s1.eval()

    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    y_true = []
    y_com = []
    com_pred = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, _, _, labels, _, _, component_name) in enumerate(tqdm(test_loader)):
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
                
                    

                
            prediction = [0 for x in prediction] 
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


#     print(' * EXP2_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance_unk(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_component_ALL_test_set_overkill_and_leakage_unk.csv')
    
    return y_com, y_pred, com_pred

def EXP_component_short(args_due , model_s1,model ,test_loader ,df ,train_component_label, val_component_label, test_component_label ,likelihood ,TH , TH_2):
    
    model.eval()
    model_s1.eval()

    df_all = df.copy()
    df_ind = df.copy()
    df_ood = df.copy()
    
    top1 = AverageMeter()
    
    y_pred = []
    y_true = []
    y_com = []
    com_pred = []
    y_true_IND = []
    y_pred_IND = []
    y_true_OOD = []
    y_pred_OOD = []
    
    with torch.no_grad():
        for idx, (images, _, _, _, labels, _, component_name) in enumerate(tqdm(test_loader)):
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
                
                    

                
            prediction = [0 for x in prediction] 
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


#     print(' * EXP2_Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    # ALL test set
    calculatePerformance_unk(df_all, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_EXP_component_ALL_test_set_overkill_and_leakage_unk.csv')
    
    return y_com, y_pred, com_pred

def HybridExpert1(y_pred_expert_detector, y_pred_expert_classifier, y_pred_expert_component, com_pred, y_true, name_label_list,  df, args_due, train_component_label, val_component_label, test_component_label):

    
    gt_label_after_unknown = []
    pred_after_unknown = []
    
    y_pred_all = []
    y_true_all = []

    new_true_label = []
    new_pred_label = []

    old_true_label = []
    old_pred_label = []
    
    prediction_IND = []
    gt_label_IND = []
    prediction_OOD = []
    gt_label_OOD = []
    
    df_ind = df.copy()
    df_ood = df.copy()
    
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]

    # train_component_label.append(21)
    # train_component_label.append(22)
    
            
    for idx, (pred_expert_detector, pred_expert_classifier, pred_expert_component, pred_com, gt_label, name) in enumerate(list(zip(y_pred_expert_detector, y_pred_expert_classifier, y_pred_expert_component, com_pred, y_true, name_label_list))):
        
        name = np.int64(name.item())
        
        if ((pred_expert_detector==0) and (pred_expert_component!=0)) or ((pred_expert_detector != 0) and (pred_expert_component==0)) or (pred_expert_component ==2) or (pred_com != name):
#             import pdb;pdb.set_trace()
            
            y_true_all.append(gt_label)
            y_pred_all.append(2)

            # if name in test_component_label :
            if pred_com in test_component_label :

                new_pred_label.append(2)
                new_true_label.append(gt_label)
                
            else :

                old_pred_label.append(2)
                old_true_label.append(gt_label)
            
            if isinstance(name, np.int64):
                df.loc[(df["component_name"] == name), 'unknown'] +=1
            else:
                df.loc[(df["component_name"] == ''.join(name)), 'unknown'] +=1

        elif (pred_expert_detector == pred_expert_component) or ((pred_expert_detector != 0) and (pred_expert_component ==1)):

            y_pred_all.append(pred_expert_classifier)
            y_true_all.append(gt_label)
            
            # if name in test_component_label :
            if pred_com in test_component_label :
                new_pred_label.append(pred_expert_classifier)
                new_true_label.append(gt_label)

            else :
                old_pred_label.append(pred_expert_classifier)
                old_true_label.append(gt_label)
            
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(pred_expert_classifier)

    defect_types = ['good',  'defect', 'unknown']
    
    cm = confusion_matrix(new_true_label, new_pred_label,labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f'actually:{label}' for label in defect_types],
                    columns=[f'predicted:{label}' for label in defect_types])
        # 可视化混淆矩阵

    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.ylabel('actually_label')
    plt.xlabel('predicted_label')
        
    # 保存图片
    plt.savefig(f'./output/{args_due.output_inference_dir}/new_com_confusion_matrix_hybridexpert1.pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    cm = confusion_matrix(old_true_label, old_pred_label,labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f'actually:{label}' for label in defect_types],
                    columns=[f'predicted:{label}' for label in defect_types])
        # 可视化混淆矩阵

    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.ylabel('actually_label')
    plt.xlabel('predicted_label')
        
    # 保存图片
    plt.savefig(f'./output/{args_due.output_inference_dir}/old_com_confusion_matrix_hybridexpert1.pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    cm = confusion_matrix(y_true_all, y_pred_all,labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f'actually:{label}' for label in defect_types],
                    columns=[f'predicted:{label}' for label in defect_types])
        # 可视化混淆矩阵

    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.ylabel('actually_label')
    plt.xlabel('predicted_label')
        
    # 保存图片
    plt.savefig(f'./output/{args_due.output_inference_dir}/all_com_confusion_matrix_hybridexpert1.pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    print("HybridExpert Accuracy: {}\n".format(100*round(accuracy_score(gt_label_after_unknown, pred_after_unknown),4)))  
    calculatePerformance_unk(df, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_HBE_ALL_test_set_overkill_and_leakage_unk.csv')

    return y_true_all, y_pred_all

def HybridExpert2(y_pred_expert_shift, y_pred_expert_broke, y_pred_expert_short, y_true_shift, y_true_broke, y_true_short, name_label_list,  df, args_due, train_component_label, val_component_label, test_component_label):

    
    gt_label_after_unknown = []
    pred_after_unknown = []
    
    y_pred_all = []
    y_true_all = []

    new_true_label = []
    new_pred_label = []

    old_true_label = []
    old_pred_label = []
    
    prediction_IND = []
    gt_label_IND = []
    prediction_OOD = []
    gt_label_OOD = []
    
    df_ind = df.copy()
    df_ood = df.copy()
    
    for name in df_ind["component_name"].value_counts().index:
        if name not in train_component_label:
            df_ind = df_ind[df_ind["component_name"] != name]
    for name in df_ood["component_name"].value_counts().index:
        if name not in test_component_label:
            df_ood = df_ood[df_ood["component_name"] != name]
         
    for idx, (pred_expert_shift, pred_expert_broke, pred_expert_short, gt_label_shift, gt_label_broke, gt_label_short, name) in enumerate(list(zip(y_pred_expert_shift, y_pred_expert_broke, y_pred_expert_short, y_true_shift, y_true_broke, y_true_short, name_label_list))):

        if (gt_label_shift==0) and (gt_label_broke==0) and (gt_label_short==0) : gt_label = 0
        elif (gt_label_shift==1) and (gt_label_broke==0) and (gt_label_short==0) : gt_label = 2
        elif (gt_label_shift==0) and (gt_label_broke==1) and (gt_label_short==0) : gt_label = 4
        elif (gt_label_shift==0) and (gt_label_broke==0) and (gt_label_short==1) : gt_label = 5
            
        if (pred_expert_shift == 0) and ((pred_expert_broke == 0) and (pred_expert_short ==0)):

            y_pred_all.append(0)
            y_true_all.append(gt_label)
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(0)
            
            if name in test_component_label :
                new_pred_label.append(0)
                new_true_label.append(gt_label)
            else :
                old_pred_label.append(0)
                old_true_label.append(gt_label)

        elif (pred_expert_shift == 1) and (pred_expert_broke != 1) and (pred_expert_short != 1):
            # and ((pred_expert_broke == 0) and (pred_expert_short ==0)):

            y_pred_all.append(2)
            y_true_all.append(gt_label)
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(2)
            
            if name in test_component_label :
                new_pred_label.append(2)
                new_true_label.append(gt_label)
            else :
                old_pred_label.append(2)
                old_true_label.append(gt_label)

        elif (pred_expert_broke == 1) and (pred_expert_shift != 1) and (pred_expert_short != 1):
        # and ((pred_expert_broke == 1) and (pred_expert_short ==0)):

            y_pred_all.append(4)
            y_true_all.append(gt_label)
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(4)
            
            if name in test_component_label :
                new_pred_label.append(4)
                new_true_label.append(gt_label)
            else :
                old_pred_label.append(4)
                old_true_label.append(gt_label)

        elif (pred_expert_short == 1) and (pred_expert_broke != 1) and (pred_expert_shift != 1) :
        # and ((pred_expert_broke == 0) and (pred_expert_short ==1)):

            y_pred_all.append(5)
            y_true_all.append(gt_label)
            gt_label_after_unknown.append(gt_label)
            pred_after_unknown.append(5)
            
            if name in test_component_label :
                new_pred_label.append(5)
                new_true_label.append(gt_label)
            else :
                old_pred_label.append(5)
                old_true_label.append(gt_label)

        else :
                
            y_true_all.append(gt_label)
            y_pred_all.append(6)
                
            if name in test_component_label :
                new_pred_label.append(6)
                new_true_label.append(gt_label)
            else :
                old_pred_label.append(6)
                old_true_label.append(gt_label)
            
    defect_types = ['good', 'missing', 'shift','stand','broke','short', 'unknown']
    # import pdb;pdb.set_trace()
    # defect_types = ['good',  'short', 'unknown']
    # true_label = ['good', 'missing', 'shift','stand','broke','short']
    # cm = confusion_matrix(new_true_label, new_pred_label,labels=[0, 1, 2, 3, 4, 5, 6])
    cm = confusion_matrix(new_true_label, new_pred_label,labels=[0, 1, 2, 3, 4, 5, 6])
    cm_df = pd.DataFrame(cm, index=[f'actually:{label}' for label in defect_types],
                    columns=[f'predicted:{label}' for label in defect_types])
        # 可视化混淆矩阵

    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.ylabel('actually_label')
    plt.xlabel('predicted_label')
        
    # 保存图片
    plt.savefig(f'./output/{args_due.output_inference_dir}/new_com_confusion_matrix_hybridexpert2.pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    cm = confusion_matrix(old_true_label, old_pred_label,labels=[0, 1, 2, 3, 4, 5, 6])
    cm_df = pd.DataFrame(cm, index=[f'actually:{label}' for label in defect_types],
                    columns=[f'predicted:{label}' for label in defect_types])
        # 可视化混淆矩阵

    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.ylabel('actually_label')
    plt.xlabel('predicted_label')
        
    # 保存图片
    plt.savefig(f'./output/{args_due.output_inference_dir}/old_com_confusion_matrix_hybridexpert2.pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    cm = confusion_matrix(y_true_all, y_pred_all,labels=[0, 1, 2, 3, 4, 5, 6])
    cm_df = pd.DataFrame(cm, index=[f'actually:{label}' for label in defect_types],
                    columns=[f'predicted:{label}' for label in defect_types])
        # 可视化混淆矩阵

    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.ylabel('actually_label')
    plt.xlabel('predicted_label')
        
    # 保存图片
    plt.savefig(f'./output/{args_due.output_inference_dir}/all_com_confusion_matrix_hybridexpert2.pdf', dpi=300, bbox_inches='tight')
        # plt.show()

    print("HybridExpert Accuracy: {}\n".format(100*round(accuracy_score(gt_label_after_unknown, pred_after_unknown),4)))  
    calculatePerformance_unk(df, file_name=f'./output/{args_due.output_inference_dir}/DUE_dataset_{args_due.output_inference_dir}_HBE_ALL_test_set_overkill_and_leakage_unk.csv')

    return y_true_all, y_pred_all


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
        "-c_shift", "--checkpoint_path_shift",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-c_broke", "--checkpoint_path_broke",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-c_short", "--checkpoint_path_short",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-mlp_c", "--mlp_checkpoint_path",
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
        "-gp_c_shift", "--gp_checkpoint_path_shift",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-gp_c_broke", "--gp_checkpoint_path_broke",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-gp_c_short", "--gp_checkpoint_path_short",
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
        default="PHISON",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','PHISON_regroup2','PHISON_regroup3','PHISON_fourcls','PHISON_shift','PHISON_broke','PHISON_short'],
        help="Pick a dataset",
    )
    parser.add_argument(
        "--dataset_shift",
        default="PHISON",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','PHISON_regroup2','PHISON_regroup3','PHISON_shift','PHISON_broke','PHISON_short'],
        help="Pick a dataset",
    )
    parser.add_argument(
        "--dataset_broke",
        default="PHISON",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','PHISON_regroup2','PHISON_regroup3','PHISON_shift','PHISON_broke','PHISON_short'],
        help="Pick a dataset",
    )
    parser.add_argument(
        "--dataset_short",
        default="PHISON",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','PHISON_regroup2','PHISON_regroup3','PHISON_shift','PHISON_broke','PHISON_short'],
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

#     # Load model's checkpoint
#     loc = 'cuda:0'
    
    args_due.dataset = 'PHISON_fourcls'
    args["checkpoint_path"] = args["checkpoint_path_shift"]
    args["gp_checkpoint_path"] = args["gp_checkpoint_path_shift"]

    # Load model's checkpoint
    loc = 'cuda:0'
    
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
#     checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")
    
    # load data
    ds = get_dataset(args_due.dataset ,args_due.random_seed , root="./data" )
#     input_size, num_classes, train_dataset, test_dataset, train_loader, train_com_loader = ds
    input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset,_ = ds
    
    # Intialize model

    print(num_classes)
    model, mlp_model, gp_model, likelihood  = set_model(args, args_due, train_com_loader , num_classes)
    
    # set TH 
    df = pd.read_csv(f'./output/CEdue_s2_shift_rd_{args_due.random_seed}/uncertainty.csv')
    TH = df['TH']
    TH_2 = df['TH_2']
    
    # Initialize dataset and dataloader
    
    if args_due.test_inference == True:
               
        val_loader, test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label = set_loader(args_due)

        # import pdb;pdb.set_trace()

        test_df_orig = test_component_name_df.copy()    
        test_component_name_df_Expert2 = test_df_orig.copy()
        test_component_name_df_HybridExpert = test_df_orig.copy()

        print("1. Testing with Discriminative Model (Expert shift detector)")  # Discriminative model
        y_true, y_pred_expert_detector = EXP_detector_shift(args_due ,model ,test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)
        print("3. Testing with DUE (Expert  shift component)")  # Discriminative model
        name_label_list, y_pred_expert_component, com_pred  = EXP_component_shift(args_due , model,gp_model ,test_loader, test_component_name_df_Expert2, train_component_label, val_component_label, test_component_label ,likelihood ,TH,TH_2)
        print("3. Testing with Hybrid Expert shift") 
        y_true_shift,y_pred_shift = HybridExpert1(y_pred_expert_detector, y_pred_expert_detector, y_pred_expert_component, com_pred, y_true, name_label_list, test_component_name_df_HybridExpert, args_due, train_component_label, val_component_label, test_component_label)

    
    # broke detector
    args["checkpoint_path"] = args["checkpoint_path_broke"]
    args["gp_checkpoint_path"] = args["gp_checkpoint_path_broke"]
    # Intialize model

    print(num_classes)
    model, mlp_model, gp_model, likelihood  = set_model(args, args_due, train_com_loader , num_classes)
    
    # set TH 
    df = pd.read_csv(f'./output/CEdue_s2_broke_rd_{args_due.random_seed}/uncertainty.csv')
    TH = df['TH']
    TH_2 = df['TH_2']

    if args_due.test_inference == True:

        print("2. Testing with Discriminative Model (Expert broke detector)")  # Discriminative model
        y_true, y_pred_expert_detector_broke = EXP_detector_broke(args_due ,model ,test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)
        print("3. Testing with DUE (Expert  broke component)")  # Discriminative model
        name_label_list, y_pred_expert_component, com_pred  = EXP_component_broke(args_due , model,gp_model ,test_loader, test_component_name_df_Expert2, train_component_label, val_component_label, test_component_label ,likelihood ,TH,TH_2)
        print("3. Testing with Hybrid Expert broke") 
        y_true_broke,y_pred_broke = HybridExpert1(y_pred_expert_detector, y_pred_expert_detector, y_pred_expert_component, com_pred, y_true, name_label_list, test_component_name_df_HybridExpert, args_due, train_component_label, val_component_label, test_component_label)

    #short detector
    args["checkpoint_path"] = args["checkpoint_path_short"]
    args["gp_checkpoint_path"] = args["gp_checkpoint_path_short"]
    # Intialize model

    print(num_classes)
    model, mlp_model, gp_model, likelihood  = set_model(args, args_due, train_com_loader , num_classes)
    
    # set TH 
    df = pd.read_csv(f'./output/CEdue_s2_short_rd_{args_due.random_seed}/uncertainty.csv')
    TH = df['TH']
    TH_2 = df['TH_2']

    if args_due.test_inference == True:

        print("3. Testing with Discriminative Model (Expert short detector)")  # Discriminative model
        y_true, y_pred_expert_detector_short = EXP_detector_short(args_due ,model ,test_loader, test_component_name_df, train_component_label, val_component_label, test_component_label)
        print("3. Testing with DUE (Expert  short component)")  # Discriminative model
        name_label_list, y_pred_expert_component, com_pred  = EXP_component_short(args_due , model,gp_model ,test_loader, test_component_name_df_Expert2, train_component_label, val_component_label, test_component_label ,likelihood ,TH,TH_2)
        print("3. Testing with Hybrid Expert short") 
        y_true_short,y_pred_short = HybridExpert1(y_pred_expert_detector, y_pred_expert_detector, y_pred_expert_component, com_pred, y_true, name_label_list, test_component_name_df_HybridExpert, args_due, train_component_label, val_component_label, test_component_label)
        
        
        print("3. Testing with Hybrid Expert") 
        HybridExpert2(y_pred_shift, y_pred_broke, y_pred_short, y_true_shift, y_true_broke, y_true_short, name_label_list, test_component_name_df_HybridExpert, args_due, train_component_label, val_component_label, test_component_label)
    else:
        print('inference calculations are not performed')
        

    end = time.time()

    print("執行時間：%f 秒" % (end - start))

    
