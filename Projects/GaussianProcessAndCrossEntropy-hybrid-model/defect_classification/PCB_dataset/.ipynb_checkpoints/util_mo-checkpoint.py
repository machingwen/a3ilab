# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import numpy as np
import pandas as pd
import json
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from pytorch_metric_learning import samplers
import sys
# sys.path.append('/root/notebooks/deepGD3_plus/DeepGD3_Plus-main/PCB_dataset/clustimage_phison/clustimage/')
# sys.path.append('/root/notebooks/Phison/defect_classification-maim/clustimage_phison/clustimage/')
# from clustimage import Clustimage
labelencoder = LabelEncoder()

# data_dir = "/home/a3ilab01/Downloads/.datasets_for_ma/"
# json_path = "/home/a3ilab01/Downloads/dataset.json"
# noisy_label_path = "/home/a3ilab01/Downloads/toRemove.txt"
# data_dir = "/root/notebooks/datasets_for_ma/"
# json_path = "/root/notebooks/Phison/dataset.json"
# noisy_label_path = "/root/notebooks/Phison/toRemove.txt"
data_dir = "/root/notebooks/GPCE/datasets_for_ma/"
json_path = "/root/notebooks/GPCE/Phison/dataset.json"
noisy_label_path = "/root/notebooks/GPCE/Phison/toRemove.txt"

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                     frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                     random_state=None):
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def split_stratified_into_train_val(df_input, stratify_colname='y',
                                     frac_train=0.8, frac_val=0.1,
                                     random_state=None):
    if frac_train + frac_val != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val))
    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
#     relative_frac_test = frac_test / (frac_val + frac_test)
#     df_val, df_test, y_val, y_test = train_test_split(df_temp,
#                                                       y_temp,
#                                                       stratify=y_temp,
#                                                       test_size=relative_frac_test,
#                                                       random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val)

    return df_train, df_val
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = 1
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        k = 1
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def set_optimizer_two(opt, model):
    params_cls = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "component_classifier" not in key:
                params_cls += [{'params': [value], 'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}]
            
    params_component = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'cls_classifier' not in key:
                params_component += [{'params': [value], 'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}]
    
    optimizer_cls = optim.SGD(params_cls,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    optimizer_component = optim.SGD(params_component,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    return optimizer_cls, optimizer_component

def save_model(model, optimizer_cls, optimizer_component, opt, epoch, save_file):
    print('==> Saving trained model...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer_cls': optimizer_cls.state_dict(),
        'optimizer_component': optimizer_component.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def save_regroup_model(model, optimizer_component, opt, epoch, save_file):
    print('==> Saving trained model...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer_component': optimizer_component.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe, transform, mode='train'):
#         self.dataframe = dataframe
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         image = self.transform(Image.open((row["file_path"])).convert('RGB'))
# #         image = self.transform(Image.open((row["file_path"])))
#         label = np.asarray(row["class"])
#         component_name = row["component_name"]
#         file_path = row["file_path"]
#         return (image, label, file_path, component_name)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, mode='train'):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self.transform(Image.open((row["file_path"])).convert('RGB'))
#         image = self.transform(Image.open((row["file_path"])))
        label = np.asarray(row["class"])
        good = np.asarray(row["good"])
        shift = np.asarray(row["shift"])
        broke = np.asarray(row["broke"])
        short = np.asarray(row["short"])
        component_name = row["component_name"]
        file_path = row["file_path"]
        return (image, label, good, shift, broke, short, file_path, component_name)

# class TsneCustomDataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe, transform, tsne=False):
#         self.dataframe = dataframe
#         self.transform = transform
#         self.tsne = tsne
#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         image = self.transform(Image.open((row["file_path"])).convert('RGB'))
# #         image = self.transform(Image.open((row["file_path"])))
#         label = np.asarray(row["class"])
#         component_name = row["component_name"]
#         component_full_name = row['component_full_name']
#         file_path = row["file_path"]
#         return (image, label, file_path, component_name, component_full_name)

def CreateDataset_ori(seed):
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5
    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]
    
    
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    print(train_df['component_name'].value_counts())
    train_df['component_name'] = labelencoder.fit_transform(train_df['component_name'])
    # Set missing, stand samples as independent components
    train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
    train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1    
    train_label_list = sorted(train_df['component_name'].value_counts().index.tolist())
    
    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print(val_df['component_name'].value_counts())
    val_label_list = val_df['component_name'].value_counts().index.tolist()
    # Set missing, stand samples as independent components
    val_df.loc[val_df['class'] == 1, ['component_name']] = 'solder_missing'
    val_df.loc[val_df['class'] == 3, ['component_name']] = 'solder_stand'
    for idx, name in enumerate(val_label_list):
        val_df.loc[val_df["component_name"] == name, "component_name"] = len(train_label_list) + idx
    
    val_df.loc[val_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
    val_df.loc[val_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print(test_df['component_name'].value_counts())
    test_label_list = test_df['component_name'].value_counts().index.tolist()
    
     # Set missing, stand samples as independent components
    test_df.loc[test_df['class'] == 1, ['component_name']] = 'solder_missing'
    test_df.loc[test_df['class'] == 3, ['component_name']] = 'solder_stand'
    
    for idx, name in enumerate(test_label_list):
        test_df.loc[test_df["component_name"] == name, "component_name"] = len(train_label_list+val_label_list) + idx
    
    test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
    test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1    
    
    print("Class distribution in Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("\nTotal dataset size: ", 
          sum(train_df['class'].value_counts().tolist()) + sum(val_df['class'].value_counts().tolist()) + sum(test_df['class'].value_counts().tolist()))
    print("Num of Images in Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    print("Num of Images of each component in Testing set: \n", test_df['component_name'].value_counts())
    
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1
    
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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)
    
    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label

    return train_df, val_df, test_df

def CreateDataset(seed ,add_test, testing=None):
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)

#     import pdb;pdb.set_trace()
    # Set missing, stand samples as independent components
    if testing is None:
        train_df.loc[train_df['class'] == 1, ['component_name']] = 35 # missing
        train_df.loc[train_df['class'] == 3, ['component_name']] = 36 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    if testing is None:
        # Set missing, stand samples as independent components
        val_df.loc[val_df['class'] == 1, ['component_name']] = 35
        val_df.loc[val_df['class'] == 3, ['component_name']] = 36
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    if testing is None:
        test_df.loc[test_df['class'] == 1, ['component_name']] = 35, #'solder_missing'
        test_df.loc[test_df['class'] == 3, ['component_name']] = 36, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
            
    
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==35)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==36)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])

    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df 



def CreateDataset_relabel_sixcls(seed, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]
    
    
    # trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(newComponent, 6)
    for i in valComponent:
        newComponent.remove(i)
    testComponent = random.sample(newComponent, 7)
    for i in testComponent:
        newComponent.remove(i)

    if seed == 11:
        valComponent = [4,8,9,12,13,14,20]
        testComponent = [2,3,5,10,11,17]

    for i in valComponent:
        trainComponent.remove(i)
    for i in testComponent:
        trainComponent.remove(i)
        
    # trainComponent.remove(1) # 元件A (樣本最多的)
    # valComponent = random.sample(trainComponent, 6)
    # for i in valComponent:
    #     trainComponent.remove(i)
    # testComponent = random.sample(trainComponent, 6)
    # for i in testComponent:
    #     trainComponent.remove(i)
    # trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # Set missing, stand samples as independent components
    if testing is None:
        train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
        train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
#     train_df.loc[train_df['class'] == 1, 'class'] = 1
#     train_df.loc[train_df['class'] == 2, 'class'] = 1
#     train_df.loc[train_df['class'] == 3, 'class'] = 1
#     train_df.loc[train_df['class'] == 4, 'class'] = 1
#     train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    if testing is None:
        # Set missing, stand samples as independent components
        val_df.loc[val_df['class'] == 1, ['component_name']] = 21
        val_df.loc[val_df['class'] == 3, ['component_name']] = 22
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
#     val_df.loc[val_df['class'] == 1, 'class'] = 1
#     val_df.loc[val_df['class'] == 2, 'class'] = 1
#     val_df.loc[val_df['class'] == 3, 'class'] = 1
#     val_df.loc[val_df['class'] == 4, 'class'] = 1
#     val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    if testing is None:
        test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
        test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
#     test_df.loc[test_df['class'] == 1, 'class'] = 1
#     test_df.loc[test_df['class'] == 2, 'class'] = 1
#     test_df.loc[test_df['class'] == 3, 'class'] = 1
#     test_df.loc[test_df['class'] == 4, 'class'] = 1
#     test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    
    # 取得new component的good sample給component classifier訓練
    for name in valComponent:
        good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
        val_df = val_df.drop(good_new_component.index)
        bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
        val_df = val_df.drop(bad_new_component_sample.index)
        train_df = pd.concat([train_df, good_new_component])
    for name in testComponent:
        good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
        test_df = test_df.drop(good_new_component.index)
        train_df = pd.concat([train_df, good_new_component])
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df

def CreateDataset_relabel_sixcls_randomsplit(seed , testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    # trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,3,5,6,10,11,16,17]
    trainDefect = [0,2,4]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = train_df['class'].isin(trainDefect)
    train_df = train_df[trainDefectDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, val_df, test_df = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    train_com_df = good_samples.copy()

    val_com_df = val_df.copy()
    good_val_samples = val_com_df.loc[val_com_df['class']==0]
    # missing_val_samples = val_com_df.loc[(val_com_df['component_name']==21)]
    # stand_val_samples = val_com_df.loc[(val_com_df['component_name']==22)]
    # val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])
    val_com_df = good_samples.copy()
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df
    # , val_com_df

def CreateDataset_relabel_fourcls(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'good'] = 0
    df.loc[df['class'] == 1, 'good'] = 1
    df.loc[df['class'] == 2, 'good'] = 1
    df.loc[df['class'] == 3, 'good'] = 1
    df.loc[df['class'] == 4, 'good'] = 1
    df.loc[df['class'] == 5, 'good'] = 1

    df.loc[df['class'] == 0, 'shift'] = 0
    df.loc[df['class'] == 1, 'shift'] = 0
    df.loc[df['class'] == 2, 'shift'] = 1
    df.loc[df['class'] == 3, 'shift'] = 0
    df.loc[df['class'] == 4, 'shift'] = 0
    df.loc[df['class'] == 5, 'shift'] = 0

    df.loc[df['class'] == 0, 'broke'] = 0
    df.loc[df['class'] == 1, 'broke'] = 0
    df.loc[df['class'] == 2, 'broke'] = 0
    df.loc[df['class'] == 3, 'broke'] = 0
    df.loc[df['class'] == 4, 'broke'] = 1
    df.loc[df['class'] == 5, 'broke'] = 0

    df.loc[df['class'] == 0, 'short'] = 0
    df.loc[df['class'] == 1, 'short'] = 0
    df.loc[df['class'] == 2, 'short'] = 0
    df.loc[df['class'] == 3, 'short'] = 0
    df.loc[df['class'] == 4, 'short'] = 0
    df.loc[df['class'] == 5, 'short'] = 1
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)

    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    # val_component_label = val_df['component_name'].value_counts().index.tolist()
    # test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_good(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 1
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)

    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_shift(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 0
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 0
    df.loc[df['class'] == 4, 'class'] = 0
    df.loc[df['class'] == 5, 'class'] = 0
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)

    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_broke(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 0
    df.loc[df['class'] == 2, 'class'] = 0
    df.loc[df['class'] == 3, 'class'] = 0
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 0
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_short(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/GPCE/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,2,3,5,6,10,11,14,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = df['class'].isin(trainDefect)
    df = df[trainDefectDatasetMask].copy()
    
#     newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]

    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 0
    df.loc[df['class'] == 2, 'class'] = 0
    df.loc[df['class'] == 3, 'class'] = 0
    df.loc[df['class'] == 4, 'class'] = 0
    df.loc[df['class'] == 5, 'class'] = 1
    
    trainComponent.remove(1)
    trainComponent.remove(6)
    
    valComponent = random.sample(trainComponent, 2)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 2)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    testComponent.append(6)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
        
    test_df = pd.concat([test_df, ind_test])

    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    # val_component_label = val_df['component_name'].value_counts().index.tolist()
    # test_component_label = test_df['component_name'].value_counts().index.tolist()

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_shift_randomsplit(seed , testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    # trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,3,5,6,10,11,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = train_df['class'].isin(trainDefect)
    train_df = train_df[trainDefectDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 0
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 0
    train_df.loc[train_df['class'] == 4, 'class'] = 0
    train_df.loc[train_df['class'] == 5, 'class'] = 0
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, val_df, test_df = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    
#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_df['class']==0]
#     # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
#     # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
#     # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
#     train_com_df = good_samples.copy()
    

#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     # missing_val_samples = val_com_df.loc[(val_com_df['component_name']==21)]
#     # stand_val_samples = val_com_df.loc[(val_com_df['component_name']==22)]
#     # val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])
#     val_com_df = good_samples.copy()
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_broke_randomsplit(seed, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    # trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,3,5,6,10,11,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = train_df['class'].isin(trainDefect)
    train_df = train_df[trainDefectDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 0
    train_df.loc[train_df['class'] == 2, 'class'] = 0
    train_df.loc[train_df['class'] == 3, 'class'] = 0
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 0
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, val_df, test_df = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    
#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_df['class']==0]
#     # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
#     # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
#     # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
#     train_com_df = good_samples.copy()
    

#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     # missing_val_samples = val_com_df.loc[(val_com_df['component_name']==21)]
#     # stand_val_samples = val_com_df.loc[(val_com_df['component_name']==22)]
#     # val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])
#     val_com_df = good_samples.copy()
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df

def CreateDataset_relabel_short_randomsplit(seed, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()
    # 將Test set從Training set中移除並重新切割資料集
    # trainComponent = df['component_name'].value_counts().index.tolist()

    trainComponent = [1,3,5,6,10,11,16,17]
    trainDefect = [0,2,4,5]
    
#     trainComponent = [1,6,10,11]
#     trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = train_df['class'].isin(trainDefect)
    train_df = train_df[trainDefectDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 0
    train_df.loc[train_df['class'] == 2, 'class'] = 0
    train_df.loc[train_df['class'] == 3, 'class'] = 0
    train_df.loc[train_df['class'] == 4, 'class'] = 0
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, val_df, test_df = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)
    
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    
#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_df['class']==0]
#     # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
#     # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
#     # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
#     train_com_df = good_samples.copy()
    

#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     # missing_val_samples = val_com_df.loc[(val_com_df['component_name']==21)]
#     # stand_val_samples = val_com_df.loc[(val_com_df['component_name']==22)]
#     # val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])
#     val_com_df = good_samples.copy()
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    # missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    # stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    # train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    train_com_df = good_samples.copy()
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_com_df = val_good_df.copy()
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, val_com_df
    
def CreateDataset_relabel(seed, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # Set missing, stand samples as independent components
    if testing is None:
        train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
        train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    if testing is None:
        # Set missing, stand samples as independent components
        val_df.loc[val_df['class'] == 1, ['component_name']] = 21
        val_df.loc[val_df['class'] == 3, ['component_name']] = 22
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    if testing is None:
        test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
        test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    
    # 取得new component的good sample給component classifier訓練
    for name in valComponent:
        good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
        val_df = val_df.drop(good_new_component.index)
        bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
        val_df = val_df.drop(bad_new_component_sample.index)
        train_df = pd.concat([train_df, good_new_component])
    for name in testComponent:
        good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
        test_df = test_df.drop(good_new_component.index)
        train_df = pd.concat([train_df, good_new_component])
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df

def CreateDataset_regroup(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # Set missing, stand samples as independent components
    if testing is None:
        train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
        train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    if testing is None:
        # Set missing, stand samples as independent components
        val_df.loc[val_df['class'] == 1, ['component_name']] = 21
        val_df.loc[val_df['class'] == 3, ['component_name']] = 22
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    if testing is None:
        test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
        test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    train_regroup_df = train_df.copy()
    good_samples = train_regroup_df.loc[train_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    train_regroup_df = good_samples
    a = Counter(train_regroup_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_regroup_df.loc[train_regroup_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_regroup_df[train_regroup_df['component_name']==i].index
            train_regroup_df=train_regroup_df.drop(df_idx)
            train_regroup_df = pd.concat([train_regroup_df, component])

#     train_good_df = train_df.loc[train_df['class']==0]
#     train_bad_df = train_df.loc[train_df['class']==1]


    train_com_df = train_regroup_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

#     train_good_df = train_df.copy()
#     train_good_df = train_good_df.loc[train_good_df['class']==0]
#     a = Counter(train_good_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >10000:
#             component = train_good_df.loc[train_good_df['component_name']==i]
#             component = component.sample(n=10000,random_state=123,axis=0)
#             df_idx = train_good_df[train_good_df['component_name']==i].index
#             train_good_df=train_good_df.drop(df_idx)
#             train_good_df = pd.concat([train_good_df, component])

#     train_bad_df = train_df.copy()
#     train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
#     a = Counter(train_bad_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >10000:
#             component = train_bad_df.loc[train_bad_df['component_name']==i]
#             component = component.sample(n=10000,random_state=123,axis=0)
#             df_idx = train_bad_df[train_bad_df['component_name']==i].index
#             train_bad_df=train_bad_df.drop(df_idx)
#             train_bad_df = pd.concat([train_bad_df, component])

#     train_df = pd.concat([train_good_df, train_bad_df])


#     val_good_df = val_df.copy()
#     val_good_df = val_good_df.loc[val_good_df['class']==0]
#     a = Counter(val_good_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = val_good_df.loc[val_good_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = val_good_df[val_good_df['component_name']==i].index
#             val_good_df=val_good_df.drop(df_idx)
#             val_good_df = pd.concat([val_good_df, component])

#     val_bad_df = val_df.copy()
#     val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
#     a = Counter(val_bad_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = val_bad_df.loc[val_bad_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = val_bad_df[val_bad_df['component_name']==i].index
#             val_bad_df=val_bad_df.drop(df_idx)
#             val_bad_df = pd.concat([val_bad_df, component])

#     val_df = pd.concat([val_good_df, val_bad_df])

    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, train_regroup_df ,df
def get_label(seed,clust ,new_group , new_group_component_name):
        
    # Load model
#     cl = Clustimage(method='pca')
#     cl.load(f'clust/{seed}_pretrain_all_clustimage_model')
    label_0=[]
    
    cn = new_group_component_name
#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(cn)
    
    result = np.where(clust.results['labels'] == new_group)
    result = result[0].tolist()
    label0=[]
    for i in result:
        label0.append(cn[i])
    counter_label = Counter(label0)
    
    for i in range(len(set(Counter_cn))):
        if counter_label[i]>= (Counter_cn[i]/2):
            label_0.append(i)

    return label_0

def CreateDataset_regroup_due(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    cl = Clustimage(method='pca')
    cl.load(f'/root/notebooks/DUE/clust/{seed}_pretrain_all_clustimage_model')
    
    missing_label = len(set(cl.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # Set missing, stand samples as independent components
#     if testing is None:
    train_df.loc[train_df['class'] == 1, ['component_name']] = 35 # missing
    train_df.loc[train_df['class'] == 3, ['component_name']] = 36 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)

#     if testing is None:
        # Set missing, stand samples as independent components
    val_df.loc[val_df['class'] == 1, ['component_name']] = 35
    val_df.loc[val_df['class'] == 3, ['component_name']] = 36
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)

#     if testing is None:
    test_df.loc[test_df['class'] == 1, ['component_name']] = 35, #'solder_missing'
    test_df.loc[test_df['class'] == 3, ['component_name']] = 36, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])
    
    ### train_component_df
    train_regroup_df = train_df.copy()
    good_samples = train_regroup_df.loc[train_df['class']==0]
    train_regroup_df = good_samples
    a = Counter(train_regroup_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_regroup_df.loc[train_regroup_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_regroup_df[train_regroup_df['component_name']==i].index
            train_regroup_df=train_regroup_df.drop(df_idx)
            train_regroup_df = pd.concat([train_regroup_df, component])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==35)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==36)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])

#     test_good_df = test_df.copy()
#     test_good_df = test_good_df.loc[test_good_df['class']==0]
#     a = Counter(test_good_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = test_good_df.loc[test_good_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = test_good_df[test_good_df['component_name']==i].index
#             test_good_df=test_good_df.drop(df_idx)
#             test_good_df = pd.concat([test_good_df, component])

#     test_bad_df = test_df.copy()
#     test_bad_df = test_bad_df.loc[test_bad_df['class']==1]
#     a = Counter(test_bad_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = test_bad_df.loc[test_bad_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = test_bad_df[test_bad_df['component_name']==i].index
#             test_bad_df=test_bad_df.drop(df_idx)
#             test_bad_df = pd.concat([test_bad_df, component])

#     test_df = pd.concat([test_good_df, test_bad_df])

    _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    
    new_group_list = list(set(cl.results['labels']))
    
    cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(cn)
    val_regroup_df = val_df.copy()
    train_cls_regroup_df = train_df.copy()
    train_com_regroup_df = train_com_df.copy()
    
    test_regroup_df = test_df.copy()
    
    for new_group in new_group_list:
        label_newgroup = get_label(seed, cl ,new_group , train_regroup_df)
        
        for i in label_newgroup:
            
            train_com_regroup_df.loc[train_com_df['component_name'] == i, ['component_name']] = new_group            
            val_regroup_df.loc[val_df['component_name'] == i, ['component_name']] = new_group
            test_regroup_df.loc[test_df['component_name'] == i, ['component_name']] = new_group
            train_cls_regroup_df.loc[train_df['component_name'] == i, ['component_name']] = new_group
            
    val_regroup_df.loc[val_regroup_df['component_name'] == 35, ['component_name']] = missing_label
    val_regroup_df.loc[val_regroup_df['component_name'] == 36, ['component_name']] = stand_label
    train_com_regroup_df.loc[train_com_regroup_df['component_name'] == 35, ['component_name']] = missing_label # missing
    train_com_regroup_df.loc[train_com_regroup_df['component_name'] == 36, ['component_name']] = stand_label
    test_regroup_df.loc[test_regroup_df['component_name'] == 35, ['component_name']] = missing_label #'solder_missing'
    test_regroup_df.loc[test_regroup_df['component_name'] == 36, ['component_name']] = stand_label
    
    print("Train component label: ")
    train_component_label = train_com_regroup_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    print("Val component label: ")
    val_component_label = val_regroup_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    print("Test component label: ")
    test_component_label = test_regroup_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
#     test_component_name=[]
#     print("Test component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in test_component_label:  
#             test_component_name.append(v)
#     print(test_component_name)
    
    print("Class distribution in Component Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))

    return train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df ,train_cls_regroup_df

def CreateDataset_regroup_due_test(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    cl = Clustimage(method='pca')
    cl.load(f'/root/notebooks/DUE/clust/{seed}_pretrain_all_clustimage_model')
    
    missing_label = len(set(cl.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}

#     df.loc[df['class'] == 1, ['component_name']] = 35 # missing
#     df.loc[df['class'] == 3, ['component_name']] = 36 # stand
#     df.loc[df['class'] == 0, 'class'] = 0
#     df.loc[df['class'] == 1, 'class'] = 1
#     df.loc[df['class'] == 2, 'class'] = 1
#     df.loc[df['class'] == 3, 'class'] = 1
#     df.loc[df['class'] == 4, 'class'] = 1
#     df.loc[df['class'] == 5, 'class'] = 1

    ### train_defect_df
#     good_df = df.copy()
#     good_df = good_df.loc[good_df['class']==0]
#     aaa = Counter(good_df['component_name'])
#     for i in range(max(aaa)):
#         if aaa[i] >10000:
#             component = good_df.loc[good_df['component_name']==i]
#             component = component.sample(n=10000,random_state=42,axis=0)
#             df_idx = good_df[good_df['component_name']==i].index
#             good_df = good_df.drop(df_idx)
#             good_df = pd.concat([good_df, component])

#     bad_df = df.copy()
#     stand_df = bad_df.loc[ ((bad_df['class']!=1) &( bad_df['class']!=0)) ]
#     bad_df = bad_df.loc[bad_df['class']==1]
#     aaa = Counter(bad_df['component_name'])

#     for i in range(max(aaa)):
#         if aaa[i] >10000:
#             component = bad_df.loc[bad_df['component_name']==i]
#             component = component.sample(n=10000,random_state=42,axis=0)
#             df_idx = bad_df[bad_df['component_name']==i].index
#             bad_df = bad_df.drop(df_idx)
#             bad_df = pd.concat([bad_df, component])

#     bad_df =  pd.concat([bad_df, stand_df])

#     df = pd.concat([good_df, bad_df])
    if testing == False:
        df.loc[df['class'] == 1, ['component_name']] = 35 # missing
        df.loc[df['class'] == 3, ['component_name']] = 36 # stand
    
    _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    
    new_group_list = list(set(cl.results['labels']))
#     new_group_list = [ i + 1 for i in new_group_list] 
    
    cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(cn)
    regroup_df = df.copy()

#     if testing == True:
#         regroup_df.loc[regroup_df['class'] == 1, ['component_name']] = 35 # missing
#         regroup_df.loc[regroup_df['class'] == 3, ['component_name']] = 36 # stand    
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, cl ,new_group , train_regroup_df)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group       
    regroup_df.loc[regroup_df['class'] == 1, ['component_name']] = missing_label
    regroup_df.loc[regroup_df['class'] == 3, ['component_name']] = stand_label
    
    regroup_df.loc[regroup_df['class'] == 0, 'class'] = 0
    regroup_df.loc[regroup_df['class'] == 1, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 2, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 3, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 4, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 5, 'class'] = 1

    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = regroup_df['component_name'].value_counts().index.tolist()
    trainComponent.remove(missing_label)
    trainComponent.remove(stand_label)
    valComponent = random.sample(trainComponent, 3)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 3)
    for i in testComponent:
        trainComponent.remove(i)

    trainComponent.append(missing_label)
    trainComponent.append(stand_label)
    
    regroup_df = df.copy()
    
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, cl ,new_group , train_regroup_df)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group
    
    regroup_df.loc[regroup_df['class'] == 0, 'class'] = 0
    regroup_df.loc[regroup_df['class'] == 1, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 2, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 3, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 4, 'class'] = 1
    regroup_df.loc[regroup_df['class'] == 5, 'class'] = 1
    
    df = regroup_df.copy()
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)

    val_df = pd.concat([val_df, ind_val])
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df = pd.concat([test_df, ind_test])
    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])


#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     missing_val_samples = val_com_df.loc[(val_com_df['component_name']==missing_label)]
#     stand_val_samples = val_com_df.loc[(val_com_df['component_name']==stand_label)]
#     val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])


#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    # import pdb;pdb.set_trace()
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df

def CreateDataset_regroup_due_2(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/DUE/clust/{seed}_pretrain_all_clustimage_model')
    
    missing_label = len(set(clust.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}

    if testing is None:
        df.loc[df['class'] == 1, ['component_name']] = 35 # missing
        df.loc[df['class'] == 3, ['component_name']] = 36 # stand
    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 1


#     _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    new_group_component_name = clust.results['filenames']
    new_group_list = list(set(clust.results['labels']))
#     new_group_list = [ i + 1 for i in new_group_list] 

#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(new_group_component_name)
    regroup_df = df.copy()
    new_group=[]
#     import pdb;pdb.set_trace()
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, clust ,new_group , new_group_component_name)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group     
#             import pdb;pdb.set_trace() 
            
    if testing is None:
        regroup_df.loc[regroup_df['component_name'] == 35, ['component_name']] = missing_label
        regroup_df.loc[regroup_df['component_name'] == 36, ['component_name']] = stand_label
    
    df = regroup_df.copy()
    
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
#     
    trainComponent.remove(missing_label)
    trainComponent.remove(stand_label)
    valComponent = random.sample(trainComponent, 3)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 3)
    for i in testComponent:
        trainComponent.remove(i)

    trainComponent.append(missing_label)
    trainComponent.append(stand_label)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    
    val_df = pd.concat([val_df, ind_val])
    
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df = pd.concat([test_df, ind_test])
   
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
            
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])


#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     missing_val_samples = val_com_df.loc[(val_com_df['component_name']==missing_label)]
#     stand_val_samples = val_com_df.loc[(val_com_df['component_name']==stand_label)]
#     val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])


#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df
def CreateDataset_regroup_due_2_seed1(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/DUE/clust/1_pretrain_all_clustimage_model')
    
    missing_label = len(set(clust.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()

    if testing is None:
        df.loc[df['class'] == 1, ['component_name']] = 35 # missing
        df.loc[df['class'] == 3, ['component_name']] = 36 # stand
    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 1


#     _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    new_group_component_name = clust.results['filenames']
    new_group_list = list(set(clust.results['labels']))
#     new_group_list = [ i + 1 for i in new_group_list] 

#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(new_group_component_name)
    regroup_df = df.copy()
    new_group=[]
#     import pdb;pdb.set_trace()
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, clust ,new_group , new_group_component_name)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group     
#         import pdb;pdb.set_trace() 
            
    if testing is None:
        regroup_df.loc[regroup_df['component_name'] == 35, ['component_name']] = missing_label
        regroup_df.loc[regroup_df['component_name'] == 36, ['component_name']] = stand_label

    df = regroup_df.copy()

    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
#     import pdb;pdb.set_trace()
    trainComponent.remove(missing_label)
    trainComponent.remove(stand_label)
    valComponent = random.sample(trainComponent, 3)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 3)
    for i in testComponent:
        trainComponent.remove(i)

    trainComponent.append(missing_label)
    trainComponent.append(stand_label)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)

    val_df = pd.concat([val_df, ind_val])
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df = pd.concat([test_df, ind_test])
    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])


#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     missing_val_samples = val_com_df.loc[(val_com_df['component_name']==missing_label)]
#     stand_val_samples = val_com_df.loc[(val_com_df['component_name']==stand_label)]
#     val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])


#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df
def CreateDataset_regroup_due_2_seed42(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/DUE/clust/42_pretrain_all_clustimage_model')
    
    missing_label = len(set(clust.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()

    if testing is None:
        df.loc[df['class'] == 1, ['component_name']] = 35 # missing
        df.loc[df['class'] == 3, ['component_name']] = 36 # stand
    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 1


#     _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    new_group_component_name = clust.results['filenames']
    new_group_list = list(set(clust.results['labels']))
#     new_group_list = [ i + 1 for i in new_group_list] 

#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(new_group_component_name)
    regroup_df = df.copy()
    new_group=[]
#     import pdb;pdb.set_trace()
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, clust ,new_group , new_group_component_name)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group     
#         import pdb;pdb.set_trace() 
            
    if testing is None:
        regroup_df.loc[regroup_df['component_name'] == 35, ['component_name']] = missing_label
        regroup_df.loc[regroup_df['component_name'] == 36, ['component_name']] = stand_label

    df = regroup_df.copy()

    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
#     import pdb;pdb.set_trace()
    trainComponent.remove(missing_label)
    trainComponent.remove(stand_label)
    valComponent = random.sample(trainComponent, 3)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 3)
    for i in testComponent:
        trainComponent.remove(i)

    trainComponent.append(missing_label)
    trainComponent.append(stand_label)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)

    val_df = pd.concat([val_df, ind_val])
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df = pd.concat([test_df, ind_test])
    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])


#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     missing_val_samples = val_com_df.loc[(val_com_df['component_name']==missing_label)]
#     stand_val_samples = val_com_df.loc[(val_com_df['component_name']==stand_label)]
#     val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])


#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df
def CreateDataset_regroup_due_2_seed1212(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/clust/1212_pretrain_all_clustimage_model')
    
    missing_label = len(set(clust.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()


#     _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    new_group_component_name = clust.results['filenames']
    new_group_list = list(set(clust.results['labels']))
#     new_group_list = [ i + 1 for i in new_group_list] 

#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(new_group_component_name)
    regroup_df = df.copy()
    new_group=[]
#     import pdb;pdb.set_trace()
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, clust ,new_group , new_group_component_name)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group     
        # import pdb;pdb.set_trace() 
    df = regroup_df.copy()

    # import pdb;pdb.set_trace()

    # 將Test set從Training set中移除並重新切割資料集

    trainComponent = df['component_name'].value_counts().index.tolist()

#     valComponent = random.sample(trainComponent, 3)
#     for i in valComponent:
#         trainComponent.remove(i)
#     testComponent = random.sample(trainComponent, 3)
#     for i in testComponent:
#         trainComponent.remove(i)
    # testComponent = [seed]
    # for i in testComponent:
    #     trainComponent.remove(i)
    # valComponent = random.sample(trainComponent, 3)
    # for i in valComponent:
    #     trainComponent.remove(i)
        
    if seed == 2:
        valComponent =[1,4,8]
        testComponent = [2,6,7]
    # elif seed == 211:
    #     valComponent =[0,5,11]
    #     testComponent = [1,7,8]
    # elif seed == 1212:
    #     valComponent =[2,8,9]
    #     testComponent = [3,7,10]
    # elif seed == 2810:
    #     valComponent =[1,4,7]
    #     testComponent = [2,8,10]
    # elif seed == 1510:
    #     valComponent = [2,7,8]
    #     testComponent = [1,5,10]
    # elif seed == 130:
    #     valComponent = [2,7,8]
    #     testComponent = [0,1,5]
    elif seed == 238:
        valComponent =[0,5,11]
        testComponent = [2,3,8]
    elif seed == 589:
        valComponent = [0,2,11]
        testComponent = [5,8,9]
    elif seed == 520:
        valComponent = [8,9,10]
        testComponent = [5,2,0]
        
    for i in valComponent:
        trainComponent.remove(i)
    for i in testComponent:
        trainComponent.remove(i)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    train_df.loc[train_df['class'] == 1, ['component_name']] = missing_label
    train_df.loc[train_df['class'] == 3, ['component_name']] = stand_label

#     train_df.loc[train_df['class'] == 0, 'class'] = 0
#     train_df.loc[train_df['class'] == 1, 'class'] = 1
#     train_df.loc[train_df['class'] == 2, 'class'] = 1
#     train_df.loc[train_df['class'] == 3, 'class'] = 1
#     train_df.loc[train_df['class'] == 4, 'class'] = 1
#     train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    
    val_df.loc[val_df['class'] == 1, ['component_name']] = missing_label
    val_df.loc[val_df['class'] == 3, ['component_name']] = stand_label

#     val_df.loc[val_df['class'] == 0, 'class'] = 0
#     val_df.loc[val_df['class'] == 1, 'class'] = 1
#     val_df.loc[val_df['class'] == 2, 'class'] = 1
#     val_df.loc[val_df['class'] == 3, 'class'] = 1
#     val_df.loc[val_df['class'] == 4, 'class'] = 1
#     val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df.loc[test_df['class'] == 1, ['component_name']] = missing_label
    test_df.loc[test_df['class'] == 3, ['component_name']] = stand_label

#     test_df.loc[test_df['class'] == 0, 'class'] = 0
#     test_df.loc[test_df['class'] == 1, 'class'] = 1
#     test_df.loc[test_df['class'] == 2, 'class'] = 1
#     test_df.loc[test_df['class'] == 3, 'class'] = 1
#     test_df.loc[test_df['class'] == 4, 'class'] = 1
#     test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

    name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
    num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
    test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

    for name in set(test_df_mapping2_label['component_name'].values):
        temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
        for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
            if k == 0:
                test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
            elif k != 0:
                try:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])


#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     missing_val_samples = val_com_df.loc[(val_com_df['component_name']==missing_label)]
#     stand_val_samples = val_com_df.loc[(val_com_df['component_name']==stand_label)]
#     val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])


#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df
def CreateDataset_regroup_due_2_fewshot(seed , add_test, nshot, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/DUE/clust/{seed}_pretrain_all_clustimage_model')
    
    missing_label = len(set(clust.results['labels']))
    stand_label = missing_label + 1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}

    if testing is None:
        df.loc[df['class'] == 1, ['component_name']] = 35 # missing
        df.loc[df['class'] == 3, ['component_name']] = 36 # stand
    df.loc[df['class'] == 0, 'class'] = 0
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 1
    df.loc[df['class'] == 5, 'class'] = 1


#     _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup(seed ,add_test)
    new_group_component_name = clust.results['filenames']
    new_group_list = list(set(clust.results['labels']))
#     new_group_list = [ i + 1 for i in new_group_list] 

#     cn = train_regroup_df['component_name'].tolist()
    Counter_cn = Counter(new_group_component_name)
    regroup_df = df.copy()
    new_group=[]
#     import pdb;pdb.set_trace()
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, clust ,new_group , new_group_component_name)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group     
#         import pdb;pdb.set_trace() 
            
    if testing is None:
        regroup_df.loc[regroup_df['component_name'] == 35, ['component_name']] = missing_label
        regroup_df.loc[regroup_df['component_name'] == 36, ['component_name']] = stand_label

    df = regroup_df.copy()

    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
#     import pdb;pdb.set_trace()
    trainComponent.remove(missing_label)
    trainComponent.remove(stand_label)
    valComponent = random.sample(trainComponent, 3)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 3)
    for i in testComponent:
        trainComponent.remove(i)

    trainComponent.append(missing_label)
    trainComponent.append(stand_label)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)

    val_df = pd.concat([val_df, ind_val])
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df = pd.concat([test_df, ind_test])
    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            
            try:
                bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)].sample(n=nshot, random_state=123) #1,5,10,25,50,100
                val_df = val_df.drop(bad_new_component_sample.index)
                train_df = pd.concat([train_df, good_new_component, bad_new_component_sample])
            except:
                train_df = pd.concat([train_df, good_new_component])

        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            
            try:
                bad_new_component_sample = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] != 0)].sample(n=nshot, random_state=123)
                test_df = test_df.drop(bad_new_component_sample.index)
                train_df = pd.concat([train_df, good_new_component, bad_new_component_sample])
            except:
                train_df = pd.concat([train_df, good_new_component])
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    a = Counter(train_com_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    a = Counter(train_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
    a = Counter(train_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    a = Counter(val_good_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
    a = Counter(val_bad_df['component_name'])
    for i in range(max(a)):
        if a[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])
            
    val_df = pd.concat([val_good_df, val_bad_df])
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df

def get_fruit2(root, seed):
    random.seed(seed)
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    train_cls_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    train_cls_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_EXP1_csv.csv')
    train_com_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_EXP2_csv.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_test_csv.csv')


# 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_regroup_df['component_name'].value_counts().index.tolist()

    if seed == 1: 
        badComponent = [0]
    elif seed == 42: 
        badComponent = [1]
    elif seed == 1212: 
        badComponent = [2]
    else :
        badComponent = random.sample(trainComponent, 1)

    trainDatasetMask = (train_cls_regroup_df['component_name'].isin(badComponent)) & (train_cls_regroup_df['class'] == 1)

#     valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)

    train_cls_regroup_df = train_cls_regroup_df[-trainDatasetMask].copy()
    

    train_df, val_df = split_stratified_into_train_val(train_cls_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)
    train_com_df, val_com_df = split_stratified_into_train_val(train_com_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)

    num_classes = len(set(train_com_regroup_df['component_name']))
#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))
    
    train_cls_dataset = CustomDataset(train_df, transform=train_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
#     test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset, train_df, val_df , test_df

def get_fruit_8(root, seed):
    
    random.seed(seed)
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

#     train_cls_transform = transforms.Compose(
#         [
#             transforms.Resize([input_size, input_size]),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ToTensor(),
#             normalize,
#         ]
#     )


    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    train_cls_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_EXP1.csv')
    train_com_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_EXP2.csv')
    val_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_val.csv')
    val_com_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_val_com.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_test.csv')
    
    # 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_df['component_name'].value_counts().index.tolist()

    badComponent = random.sample(trainComponent, 2)

    trainDatasetMask = (train_cls_df['component_name'].isin(badComponent)) & (train_cls_df['class'] == 1)
    
    valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)
    
    train_cls_df = train_cls_df[-trainDatasetMask].copy()
    val_df = val_df[-valDatasetMask].copy()
#     train_df, val_df = split_stratified_into_train_val(train_cls_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)
#     train_com_df, val_com_df = split_stratified_into_train_val(train_com_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)

    num_classes = len(set(train_com_df['component_name']))
#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))
    
    train_cls_dataset = CustomDataset(train_cls_df, transform=train_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
#     test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset, train_cls_df, val_df, test_df

def get_mvtecad_tex(root, seed):
    
    random.seed(seed)
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

#     train_cls_transform = transforms.Compose(
#         [
#             transforms.Resize([input_size, input_size]),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ToTensor(),
#             normalize,
#         ]
#     )


    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    train_cls_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_exp1.csv')
    train_com_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_exp2.csv')
    val_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_val_exp1.csv')
    val_com_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_val_exp2.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_val_exp1.csv')
    
        # 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_df['component_name'].value_counts().index.tolist()

    badComponent = random.sample(trainComponent, 1)

    trainDatasetMask = (train_cls_df['component_name'].isin(badComponent)) & (train_cls_df['class'] == 1)
    
    valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)
    
    train_cls_df = train_cls_df[-trainDatasetMask].copy()
    val_df = val_df[-valDatasetMask].copy()


#     train_df, val_df = split_stratified_into_train_val(train_cls_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)
#     train_com_df, val_com_df = split_stratified_into_train_val(train_com_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)

    num_classes = len(set(train_com_df['component_name']))
#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))
    
    train_cls_dataset = CustomDataset(train_cls_df, transform=train_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)  
    
    test_dataset = CustomDataset(val_df, transform=test_transform)
    test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
#     test_com_dataset = CustomDataset(val_com_df, transform=test_transform)
    
    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset, train_cls_df, val_df, test_df

def CreateTSNEdataset_mvtecad_tex(seed , tsne=False):
    
    add_test=True
#     train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_cls_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_exp1.csv')
    train_com_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_exp2.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_texture_dataset_csv/mvtecad_tex_val_exp1.csv')
    
    random.seed(seed)
    # 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_df['component_name'].value_counts().index.tolist()

    badComponent = random.sample(trainComponent, 1)

    trainDatasetMask = (train_cls_df['component_name'].isin(badComponent)) & (train_cls_df['class'] == 1)

#     valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)

    train_cls_df = train_cls_df[-trainDatasetMask].copy()
#     val_df = val_df[-valDatasetMask].copy()
    
    train_cls_df['component_full_name'] = train_cls_df['component_name']
    train_com_df['component_full_name'] = train_com_df['component_name']
#     val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----

#     train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=4, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=4, random_state=123))
#     import pdb;pdb.set_trace()
#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_cls_df, transform=val_transform)
    train_com_dataset = TsneCustomDataset(train_com_df, transform=val_transform)
#     val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     validation_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    train_com_loader = torch.utils.data.DataLoader(
        train_com_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    return training_loader, train_com_loader, test_loader

def get_mvtecad(root, seed):
    input_size = 224
#     num_classes = 23
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )    
    
    test_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),  
        normalize])
    
    train_cls_regroup_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_dataset_csv/mvtecad_dataset_EXP1.csv')
    train_com_regroup_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_dataset_csv/mvtecad_dataset_EXP2.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_dataset_csv/mvtecad_dataset_test.csv')
    

    train_df, val_df = split_stratified_into_train_val(train_cls_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)
    train_com_df, val_com_df = split_stratified_into_train_val(train_com_regroup_df, stratify_colname='component_name', frac_train=0.7, frac_val=0.3, random_state=seed)

    num_classes = len(set(train_com_regroup_df['component_name']))
#     train_cls_regroup_df = train_cls_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_regroup_df = train_com_regroup_df.sample(frac=0.01, random_state=123)
#     val_regroup_df = val_regroup_df.sample(frac=0.01, random_state=123)
#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=123))
    
    train_cls_dataset = CustomDataset(train_df, transform=train_transform)    
    train_com_dataset = CustomDataset(train_com_df, transform=train_transform)  
    
    val_dataset = CustomDataset(val_df, transform=test_transform)
    val_com_dataset = CustomDataset(val_com_df, transform=test_transform)

#     test_dataset = CustomDataset(test_df, transform=test_transform)

#     test_com_dataset = CustomDataset(val_com_df, transform=test_transform)

    per_component_num = 128 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 128 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=128, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=128, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, val_dataset ,train_cls_dataset, train_com_dataset, val_com_dataset, train_df, val_df , test_df


def get_SVHN(root):
    input_size = 32
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


# def CreateDataset_regroup_kmeans(seed , add_test, testing=None):
#     # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
#     random.seed(seed)
#     print('==> Preparing data..')
#     dataset_info = json.load(open(json_path, "r"))
#     df = pd.DataFrame.from_dict(dataset_info, orient="index")
#     df['file_path'] = df.index
#     df["file_path"] = data_dir + df["file_path"].astype(str)

#     # 分成6個class
#     df.loc[df['class'] == "good", 'class'] = 0
#     df.loc[df['class'] == "missing", 'class'] = 1
#     df.loc[df['class'] == "shift", 'class'] = 2
#     df.loc[df['class'] == "stand", 'class'] = 3
#     df.loc[df['class'] == "broke", 'class'] = 4
#     df.loc[df['class'] == "short", 'class'] = 5    
#     # 移除資料集中的Label Noise   
#     unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
#     df = df[~df.file_path.isin(unwantedData)]    

#     df['component_name'] = labelencoder.fit_transform(df['component_name'])
#     component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
#     component_name_list = [key for key, _ in component_name_counter.most_common()]
#     component_label_list = df['component_name'].value_counts().index.tolist()
#     component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     # 將Test set從Training set中移除並重新切割資料集
#     trainComponent = df['component_name'].value_counts().index.tolist()
#     trainComponent.remove(1) # 元件A (樣本最多的)
#     valComponent = random.sample(trainComponent, 6)
#     for i in valComponent:
#         trainComponent.remove(i)
#     testComponent = random.sample(trainComponent, 6)
#     for i in testComponent:
#         trainComponent.remove(i)
#     trainComponent.append(1)

#     trainDatasetMask = df['component_name'].isin(trainComponent)
#     train_df = df[trainDatasetMask].copy()

#     print("Train component label: ")
#     train_component_label = train_df['component_name'].value_counts().index.tolist()
#     print(train_component_label)
#     train_component_name=[]
#     print("Train component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in train_component_label:  
#             train_component_name.append(v)
#     print(train_component_name)

#     # Set missing, stand samples as independent components
#     if testing is None:
#         train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
#         train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
#     train_df.loc[train_df['class'] == 0, 'class'] = 0
#     train_df.loc[train_df['class'] == 1, 'class'] = 1
#     train_df.loc[train_df['class'] == 2, 'class'] = 1
#     train_df.loc[train_df['class'] == 3, 'class'] = 1
#     train_df.loc[train_df['class'] == 4, 'class'] = 1
#     train_df.loc[train_df['class'] == 5, 'class'] = 1

#     # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
#     train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

#     valDatasetMask = df['component_name'].isin(valComponent)
#     val_df = df[valDatasetMask].copy()
#     print("Val component label: ")
#     val_component_label = val_df['component_name'].value_counts().index.tolist()
#     print(val_component_label)
#     val_component_name=[]
#     print("Val component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in val_component_label:  
#             val_component_name.append(v)
#     print(val_component_name)

#     if testing is None:
#         # Set missing, stand samples as independent components
#         val_df.loc[val_df['class'] == 1, ['component_name']] = 21
#         val_df.loc[val_df['class'] == 3, ['component_name']] = 22
#     # 分成2個class (Good and Bad)
#     val_df.loc[val_df['class'] == 0, 'class'] = 0
#     val_df.loc[val_df['class'] == 1, 'class'] = 1
#     val_df.loc[val_df['class'] == 2, 'class'] = 1
#     val_df.loc[val_df['class'] == 3, 'class'] = 1
#     val_df.loc[val_df['class'] == 4, 'class'] = 1
#     val_df.loc[val_df['class'] == 5, 'class'] = 1
#     val_df = pd.concat([val_df, ind_val])
#     testDatasetMask = df['component_name'].isin(testComponent)
#     test_df = df[testDatasetMask].copy()
#     print("Test component label: ")
#     test_component_label = test_df['component_name'].value_counts().index.tolist()
#     print(test_component_label)
#     test_component_name=[]
#     print("Test component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in test_component_label:  
#             test_component_name.append(v)
#     print(test_component_name)

#     if testing is None:
#         test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
#         test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
#     test_df.loc[test_df['class'] == 0, 'class'] = 0
#     test_df.loc[test_df['class'] == 1, 'class'] = 1
#     test_df.loc[test_df['class'] == 2, 'class'] = 1
#     test_df.loc[test_df['class'] == 3, 'class'] = 1
#     test_df.loc[test_df['class'] == 4, 'class'] = 1
#     test_df.loc[test_df['class'] == 5, 'class'] = 1
#     test_df = pd.concat([test_df, ind_test])

#     with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
#         f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
#                 'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
#                 'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
#                )
#     # 用來產生overkill和leakage數值的dataframe    
#     test_df_mapping2_label = test_df.copy()
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

#     name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
#     num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
#     test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

#     for name in set(test_df_mapping2_label['component_name'].values):
#         temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
#         for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
#             if k == 0:
#                 test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
#             elif k ==1:
#                 try:
#                     test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
#                 except:
#                     print(f"{name} only contains bad label.")
#                     test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
#     test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
#     test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
#     test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
#     col = {'overkill': 0, 'leakage': 0}
#     test_component_name_df = test_component_name_df.assign(**col)

#     test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
#     print('add_test:',add_test)
#     if add_test == True:
#         # 取得new component的good sample給component classifier訓練
#         for name in valComponent:
#             good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
#             val_df = val_df.drop(good_new_component.index)
#             bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
#             val_df = val_df.drop(bad_new_component_sample.index)
#             train_df = pd.concat([train_df, good_new_component])
#         for name in testComponent:
#             good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
#             test_df = test_df.drop(good_new_component.index)
#             train_df = pd.concat([train_df, good_new_component])

#     train_regroup_df = train_df.copy()
#     good_samples = train_regroup_df.loc[train_df['class']==0]
# #     missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
# #     stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
# #     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
#     train_regroup_df = good_samples
#     a = Counter(train_regroup_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >1000:
#             component = train_regroup_df.loc[train_regroup_df['component_name']==i]
#             component = component.sample(n=1000,random_state=123,axis=0)
#             df_idx = train_regroup_df[train_regroup_df['component_name']==i].index
#             train_regroup_df=train_regroup_df.drop(df_idx)
#             train_regroup_df = pd.concat([train_regroup_df, component])

# #     train_good_df = train_df.loc[train_df['class']==0]
# #     train_bad_df = train_df.loc[train_df['class']==1]


#     train_com_df = train_regroup_df.copy()
#     good_samples = train_com_df.loc[train_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

# #     train_good_df = train_df.copy()
# #     train_good_df = train_good_df.loc[train_good_df['class']==0]
# #     a = Counter(train_good_df['component_name'])
# #     for i in range(max(a)):
# #         if a[i] >10000:
# #             component = train_good_df.loc[train_good_df['component_name']==i]
# #             component = component.sample(n=10000,random_state=123,axis=0)
# #             df_idx = train_good_df[train_good_df['component_name']==i].index
# #             train_good_df=train_good_df.drop(df_idx)
# #             train_good_df = pd.concat([train_good_df, component])

# #     train_bad_df = train_df.copy()
# #     train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
# #     a = Counter(train_bad_df['component_name'])
# #     for i in range(max(a)):
# #         if a[i] >10000:
# #             component = train_bad_df.loc[train_bad_df['component_name']==i]
# #             component = component.sample(n=10000,random_state=123,axis=0)
# #             df_idx = train_bad_df[train_bad_df['component_name']==i].index
# #             train_bad_df=train_bad_df.drop(df_idx)
# #             train_bad_df = pd.concat([train_bad_df, component])

# #     train_df = pd.concat([train_good_df, train_bad_df])


# #     val_good_df = val_df.copy()
# #     val_good_df = val_good_df.loc[val_good_df['class']==0]
# #     a = Counter(val_good_df['component_name'])
# #     for i in range(max(a)):
# #         if a[i] >5000:
# #             component = val_good_df.loc[val_good_df['component_name']==i]
# #             component = component.sample(n=5000,random_state=123,axis=0)
# #             df_idx = val_good_df[val_good_df['component_name']==i].index
# #             val_good_df=val_good_df.drop(df_idx)
# #             val_good_df = pd.concat([val_good_df, component])

# #     val_bad_df = val_df.copy()
# #     val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
# #     a = Counter(val_bad_df['component_name'])
# #     for i in range(max(a)):
# #         if a[i] >5000:
# #             component = val_bad_df.loc[val_bad_df['component_name']==i]
# #             component = component.sample(n=5000,random_state=123,axis=0)
# #             df_idx = val_bad_df[val_bad_df['component_name']==i].index
# #             val_bad_df=val_bad_df.drop(df_idx)
# #             val_bad_df = pd.concat([val_bad_df, component])

# #     val_df = pd.concat([val_good_df, val_bad_df])

#     print("Class distribution in Component Training set:")
#     print(train_df['class'].value_counts())
#     print("\nClass distribution in Val set:")
#     print(val_df['class'].value_counts())
#     print("\nClass distribution in Testing set:")
#     print(test_df['class'].value_counts())
#     print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
#     print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
#     print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
#     return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df, train_regroup_df ,df
# def CreateDataset_regroup_due_kmeans(seed , add_test, testing=None):
#     # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
#     random.seed(seed)
#     print('==> Preparing data..')
#     dataset_info = json.load(open(json_path, "r"))
#     df = pd.DataFrame.from_dict(dataset_info, orient="index")
#     df['file_path'] = df.index
#     df["file_path"] = data_dir + df["file_path"].astype(str)

#     # Load model
#     cl = Clustimage(method='pca')
#     cl.load(f'/root/notebooks/DUE/clust/{seed}_pretrain_all_clustimage_model')

#     missing_label = len(set(cl.results['labels']))
#     stand_label = missing_label + 1

#     # 分成6個class
#     df.loc[df['class'] == "good", 'class'] = 0
#     df.loc[df['class'] == "missing", 'class'] = 1
#     df.loc[df['class'] == "shift", 'class'] = 2
#     df.loc[df['class'] == "stand", 'class'] = 3
#     df.loc[df['class'] == "broke", 'class'] = 4
#     df.loc[df['class'] == "short", 'class'] = 5    
#     # 移除資料集中的Label Noise   
#     unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
#     df = df[~df.file_path.isin(unwantedData)]    

#     df['component_name'] = labelencoder.fit_transform(df['component_name'])
#     component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
#     component_name_list = [key for key, _ in component_name_counter.most_common()]
#     component_label_list = df['component_name'].value_counts().index.tolist()
#     component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     # 將Test set從Training set中移除並重新切割資料集
#     trainComponent = df['component_name'].value_counts().index.tolist()
#     trainComponent.remove(1) # 元件A (樣本最多的)
#     valComponent = random.sample(trainComponent, 6)
#     for i in valComponent:
#         trainComponent.remove(i)
#     testComponent = random.sample(trainComponent, 6)
#     for i in testComponent:
#         trainComponent.remove(i)
#     trainComponent.append(1)

#     trainDatasetMask = df['component_name'].isin(trainComponent)
#     train_df = df[trainDatasetMask].copy()

#     print("Train component label: ")
#     train_component_label = train_df['component_name'].value_counts().index.tolist()
#     print(train_component_label)
#     train_component_name=[]
#     print("Train component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in train_component_label:  
#             train_component_name.append(v)
#     print(train_component_name)

#     # Set missing, stand samples as independent components
# #     if testing is None:
#     train_df.loc[train_df['class'] == 1, ['component_name']] = 35 # missing
#     train_df.loc[train_df['class'] == 3, ['component_name']] = 36 # stand
#     train_df.loc[train_df['class'] == 0, 'class'] = 0
#     train_df.loc[train_df['class'] == 1, 'class'] = 1
#     train_df.loc[train_df['class'] == 2, 'class'] = 1
#     train_df.loc[train_df['class'] == 3, 'class'] = 1
#     train_df.loc[train_df['class'] == 4, 'class'] = 1
#     train_df.loc[train_df['class'] == 5, 'class'] = 1

#     # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
#     train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

#     valDatasetMask = df['component_name'].isin(valComponent)
#     val_df = df[valDatasetMask].copy()
#     print("Val component label: ")
#     val_component_label = val_df['component_name'].value_counts().index.tolist()
#     print(val_component_label)
#     val_component_name=[]
#     print("Val component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in val_component_label:  
#             val_component_name.append(v)
#     print(val_component_name)

# #     if testing is None:
#         # Set missing, stand samples as independent components
#     val_df.loc[val_df['class'] == 1, ['component_name']] = 35
#     val_df.loc[val_df['class'] == 3, ['component_name']] = 36
#     # 分成2個class (Good and Bad)
#     val_df.loc[val_df['class'] == 0, 'class'] = 0
#     val_df.loc[val_df['class'] == 1, 'class'] = 1
#     val_df.loc[val_df['class'] == 2, 'class'] = 1
#     val_df.loc[val_df['class'] == 3, 'class'] = 1
#     val_df.loc[val_df['class'] == 4, 'class'] = 1
#     val_df.loc[val_df['class'] == 5, 'class'] = 1
#     val_df = pd.concat([val_df, ind_val])
#     testDatasetMask = df['component_name'].isin(testComponent)
#     test_df = df[testDatasetMask].copy()
#     print("Test component label: ")
#     test_component_label = test_df['component_name'].value_counts().index.tolist()
#     print(test_component_label)
#     test_component_name=[]
#     print("Test component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in test_component_label:  
#             test_component_name.append(v)
#     print(test_component_name)

# #     if testing is None:
#     test_df.loc[test_df['class'] == 1, ['component_name']] = 35, #'solder_missing'
#     test_df.loc[test_df['class'] == 3, ['component_name']] = 36, #'solder_stand'
#     test_df.loc[test_df['class'] == 0, 'class'] = 0
#     test_df.loc[test_df['class'] == 1, 'class'] = 1
#     test_df.loc[test_df['class'] == 2, 'class'] = 1
#     test_df.loc[test_df['class'] == 3, 'class'] = 1
#     test_df.loc[test_df['class'] == 4, 'class'] = 1
#     test_df.loc[test_df['class'] == 5, 'class'] = 1
#     test_df = pd.concat([test_df, ind_test])

#     with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
#         f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
#                 'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
#                 'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
#                )
#     # 用來產生overkill和leakage數值的dataframe    
#     test_df_mapping2_label = test_df.copy()
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

#     name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
#     num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
#     test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

#     for name in set(test_df_mapping2_label['component_name'].values):
#         temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
#         for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
#             if k == 0:
#                 test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
#             elif k ==1:
#                 try:
#                     test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
#                 except:
#                     print(f"{name} only contains bad label.")
#                     test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
#     test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
#     test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
#     test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
#     col = {'overkill': 0, 'leakage': 0}
#     test_component_name_df = test_component_name_df.assign(**col)

#     test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
#     print('add_test:',add_test)
#     if add_test == True:
#         # 取得new component的good sample給component classifier訓練
#         for name in valComponent:
#             good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
#             val_df = val_df.drop(good_new_component.index)
#             bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
#             val_df = val_df.drop(bad_new_component_sample.index)
#             train_df = pd.concat([train_df, good_new_component])
#         for name in testComponent:
#             good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
#             test_df = test_df.drop(good_new_component.index)
#             train_df = pd.concat([train_df, good_new_component])

#     ### train_component_df

# #     train_regroup_df = train_df.copy()
# #     good_samples = train_regroup_df.loc[train_df['class']==0]
# #     train_regroup_df = good_samples
# #     a = Counter(train_regroup_df['component_name'])
# #     for i in range(max(a)):
# #         if a[i] >10000:
# #             component = train_regroup_df.loc[train_regroup_df['component_name']==i]
# #             component = component.sample(n=10000,random_state=123,axis=0)
# #             df_idx = train_regroup_df[train_regroup_df['component_name']==i].index
# #             train_regroup_df=train_regroup_df.drop(df_idx)
# #             train_regroup_df = pd.concat([train_regroup_df, component])

#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==35)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==36)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
#     a = Counter(train_com_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >10000:
#             component = train_com_df.loc[train_com_df['component_name']==i]
#             component = component.sample(n=10000,random_state=123,axis=0)
#             df_idx = train_com_df[train_com_df['component_name']==i].index
#             train_com_df=train_com_df.drop(df_idx)
#             train_com_df = pd.concat([train_com_df, component])


#     ### train_defect_df
#     train_good_df = train_df.copy()
#     train_good_df = train_good_df.loc[train_good_df['class']==0]
#     a = Counter(train_good_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >10000:
#             component = train_good_df.loc[train_good_df['component_name']==i]
#             component = component.sample(n=10000,random_state=123,axis=0)
#             df_idx = train_good_df[train_good_df['component_name']==i].index
#             train_good_df=train_good_df.drop(df_idx)
#             train_good_df = pd.concat([train_good_df, component])

#     train_bad_df = train_df.copy()
#     train_bad_df = train_bad_df.loc[train_bad_df['class']==1]
#     a = Counter(train_bad_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >10000:
#             component = train_bad_df.loc[train_bad_df['component_name']==i]
#             component = component.sample(n=10000,random_state=123,axis=0)
#             df_idx = train_bad_df[train_bad_df['component_name']==i].index
#             train_bad_df=train_bad_df.drop(df_idx)
#             train_bad_df = pd.concat([train_bad_df, component])

#     train_df = pd.concat([train_good_df, train_bad_df])

#     ### val_df
#     val_good_df = val_df.copy()
#     val_good_df = val_good_df.loc[val_good_df['class']==0]
#     a = Counter(val_good_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = val_good_df.loc[val_good_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = val_good_df[val_good_df['component_name']==i].index
#             val_good_df=val_good_df.drop(df_idx)
#             val_good_df = pd.concat([val_good_df, component])

#     val_bad_df = val_df.copy()
#     val_bad_df = val_bad_df.loc[val_bad_df['class']==1]
#     a = Counter(val_bad_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = val_bad_df.loc[val_bad_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = val_bad_df[val_bad_df['component_name']==i].index
#             val_bad_df=val_bad_df.drop(df_idx)
#             val_bad_df = pd.concat([val_bad_df, component])

#     val_df = pd.concat([val_good_df, val_bad_df])

#     test_good_df = test_df.copy()
#     test_good_df = test_good_df.loc[test_good_df['class']==0]
#     a = Counter(test_good_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = test_good_df.loc[test_good_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = test_good_df[test_good_df['component_name']==i].index
#             test_good_df=test_good_df.drop(df_idx)
#             test_good_df = pd.concat([test_good_df, component])

#     test_bad_df = test_df.copy()
#     test_bad_df = test_bad_df.loc[test_bad_df['class']==1]
#     a = Counter(test_bad_df['component_name'])
#     for i in range(max(a)):
#         if a[i] >5000:
#             component = test_bad_df.loc[test_bad_df['component_name']==i]
#             component = component.sample(n=5000,random_state=123,axis=0)
#             df_idx = test_bad_df[test_bad_df['component_name']==i].index
#             test_bad_df=test_bad_df.drop(df_idx)
#             test_bad_df = pd.concat([test_bad_df, component])

#     test_df = pd.concat([test_good_df, test_bad_df])

#     _, _, _, _, _, _, _, train_regroup_df, _ = CreateDataset_regroup_kmeans(seed ,add_test)

#     new_group_list = list(set(cl.results['labels']))
# #     new_group_list = [ i + 1 for i in new_group_list] 

#     cn = train_regroup_df['component_name'].tolist()
#     Counter_cn = Counter(cn)
#     val_regroup_df = val_df.copy()
#     train_cls_regroup_df = train_df.copy()
#     train_com_regroup_df = train_com_df.copy()

#     test_regroup_df = test_df.copy()
# #     import pdb;pdb.set_trace()
#     for new_group in new_group_list:
#         label_newgroup = get_label(seed, cl ,new_group , train_regroup_df)

#         for i in label_newgroup:

#             train_com_regroup_df.loc[train_com_df['component_name'] == i, ['component_name']] = new_group+1            
#             val_regroup_df.loc[val_df['component_name'] == i, ['component_name']] = new_group+1
#             test_regroup_df.loc[test_df['component_name'] == i, ['component_name']] = new_group+1
#             train_cls_regroup_df.loc[train_df['component_name'] == i, ['component_name']] = new_group+1

# ### regroup


#     val_regroup_df.loc[val_regroup_df['component_name'] == 35, ['component_name']] = missing_label
#     val_regroup_df.loc[val_regroup_df['component_name'] == 36, ['component_name']] = stand_label
#     train_com_regroup_df.loc[train_com_regroup_df['component_name'] == 35, ['component_name']] = missing_label # missing
#     train_com_regroup_df.loc[train_com_regroup_df['component_name'] == 36, ['component_name']] = stand_label
#     test_regroup_df.loc[test_regroup_df['component_name'] == 35, ['component_name']] = missing_label #'solder_missing'
#     test_regroup_df.loc[test_regroup_df['component_name'] == 36, ['component_name']] = stand_label

# #     import pdb;pdb.set_trace()
#     print("Class distribution in Component Training set:")
#     print(train_df['class'].value_counts())
#     print("\nClass distribution in Val set:")
#     print(val_df['class'].value_counts())
#     print("\nClass distribution in Testing set:")
#     print(test_df['class'].value_counts())
#     print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
#     print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
#     print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))


#     return train_cls_regroup_df, val_regroup_df, test_regroup_df, train_component_label, val_component_label, test_component_label, train_com_regroup_df


# def CreateDataset_relabel_for_each_component(seed, component_name):
#     train_df, val_df, _, _, _,_= CreateDataset_relabel(seed)    
#     train_df = train_df.loc[(train_df['component_name'] == component_name)]
#     print("Class distribution in Training set:")
#     print(train_df['class'].value_counts())
#     print("Num of Images in Training set: ", sum(train_df['class'].value_counts().tolist()))
#     return train_df
# def CreateDataset_relabel_for_each_component(seed, component_name, split='train', testing=None):
#     if split == "train":
#         df, _, _, _, _, _, _ = CreateDataset_relabel(seed, testing=True)    
#     if split == "val":
#         _, df, _, _, _, _, _ = CreateDataset_relabel(seed, testing=True)    
#     if split == "test":
#         _, _, df, _, _, _, _ = CreateDataset_relabel(seed, testing=True)    

#     df = df.loc[(df['component_name'] == component_name)]
#     print("Class distribution in Test set:")
#     print(df['class'].value_counts())
#     print("Num of Images in Test set: ", sum(df['class'].value_counts().tolist()))
#     return df

def CreateDataset_for_each_component_regroup(seed, train_val_df, component_name, split='train', testing=None):
    
    
    train_val_df = train_val_df.loc[(train_val_df['component_name'] == component_name)]
    print("Class distribution in Test set:")
    print(train_val_df['class'].value_counts())
    print("Num of Images in Test set: ", sum(train_val_df['class'].value_counts().tolist()))
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_val_dataset = CustomDataset(train_val_df, transform=val_transform)

    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_val_loader

def CreateDataset_relabel_for_each_component(seed, component_name, split='train', testing=None):
    
    
    train_val_df = train_val_df.loc[(train_val_df['component_name'] == component_name)]
    print("Class distribution in Test set:")
    print(train_val_df['class'].value_counts())
    print("Num of Images in Test set: ", sum(train_val_df['class'].value_counts().tolist()))
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_val_dataset = CustomDataset(train_val_df, transform=val_transform)

    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_val_loader

def CreateDataset_relabel_for_each_component_goodonly(seed, component_name, split='train', testing=None):
    train_df, val_df, _, _, _, _, _ = CreateDataset_relabel(seed, testing=None)   

    train_val_df = pd.concat([train_df, val_df])
    
    train_val_df = train_val_df.loc[train_val_df['class']==0]
    
    train_val_df = train_val_df.loc[(train_val_df['component_name'] == component_name)]
    print("Class distribution in Test set:")
    print(train_val_df['class'].value_counts())
    print("Num of Images in Test set: ", sum(train_val_df['class'].value_counts().tolist()))
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_val_dataset = CustomDataset(train_val_df, transform=val_transform)

    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_val_loader


def CreateDataset_for_each_component(seed, component_name, split='train', testing=None):
    add_test=True
    train_df, val_df, _, _, _, _, _ = CreateDataset(seed, add_test)    

    train_val_df = pd.concat([train_df, val_df])
    
    train_val_df = train_val_df.loc[(train_val_df['component_name'] == component_name)]
    print("Class distribution in Test set:")
    print(train_val_df['class'].value_counts())
    print("Num of Images in Test set: ", sum(train_val_df['class'].value_counts().tolist()))
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_val_dataset = CustomDataset(train_val_df, transform=val_transform)

    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_val_loader

def CreateTSNEdataset_regroup(seed , tsne=False):
    
    add_test=True
    train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----
    
    train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
    
    train_df.loc[train_df['class'] == 1, ['component_name']] = 13
        
    train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

    val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader

def CreateTSNEdataset_regroup_1212(seed , tsne=False):
    
    add_test=True
    train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_seed1212(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----
    
    train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
    
    train_df.loc[train_df['class'] == 1, ['component_name']] = 13
        
    train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

    val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=5, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=5, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader

def CreateTSNEdataset_regroup_fewshot(seed, nshot , tsne=False):
    
    add_test=True
    train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_fewshot(seed, add_test, nshot)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----
    
    train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
        
    train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

    val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader

def CreateTSNEdataset_regroup_six(seed , tsne=False):
    
    add_test=True
    # train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2_sixcls(seed, add_test)
    # train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_relabel_sixcls(seed, add_test)
    train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_relabel_sixcls_randomsplit(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----
    
    train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
        
    train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

    val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))
#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))
#     import pdb;pdb.set_trace()
    test_df_bad1 = test_df.loc[test_df["class"]==1].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))
    test_df_bad2 = test_df.loc[test_df["class"]==2].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))
    test_df_bad3 = test_df.loc[test_df["class"]==3].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))
    test_df_bad4 = test_df.loc[test_df["class"]==4].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))
    test_df_bad5 = test_df.loc[test_df["class"]==5].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df = pd.concat([test_df_good, test_df_bad])
    test_df = pd.concat([test_df_good, test_df_bad1 ,test_df_bad2 ,test_df_bad3, test_df_bad4, test_df_bad5])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader

def CreateDataset_regroup_due_2_sixcls(seed , add_test, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    dataset_info = json.load(open(json_path, "r"))
    df = pd.DataFrame.from_dict(dataset_info, orient="index")
    df['file_path'] = df.index
    df["file_path"] = data_dir + df["file_path"].astype(str)
    
    # Load model
    clust = Clustimage(method='pca')
    clust.load(f'/root/notebooks/clust/1212_pretrain_all_clustimage_model')
    
    missing_label = len(set(clust.results['labels']))
    stand_label = missing_label + 1
    shift_label = stand_label +1
    short_label = shift_label +1
    broke_label = short_label +1
    
    # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]    
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
#     import pdb;pdb.set_trace()


#     df.loc[df['class'] == 1, ['component_name']] = 35 # missing
#     df.loc[df['class'] == 3, ['component_name']] = 36 # stand
#     df.loc[df['class'] == 2, ['component_name']] = 37 # shift
#     df.loc[df['class'] == 5, ['component_name']] = 38 # short
#     df.loc[df['class'] == 4, ['component_name']] = 39 # broken

    new_group_component_name = clust.results['filenames']
    new_group_list = list(set(clust.results['labels']))

    Counter_cn = Counter(new_group_component_name)
    regroup_df = df.copy()
    new_group=[]
#     import pdb;pdb.set_trace()
    ### regroup
    for new_group in new_group_list:
        label_newgroup = get_label(seed, clust ,new_group , new_group_component_name)
        
        for i in label_newgroup:
            
            regroup_df.loc[df['component_name'] == i, ['component_name']] = new_group     

    df = regroup_df.copy()

    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    import pdb;pdb.set_trace()

    valComponent = random.sample(trainComponent, 3)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 3)
    for i in testComponent:
        trainComponent.remove(i)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)

    train_df.loc[train_df['class'] == 1, ['component_name']] = missing_label
    train_df.loc[train_df['class'] == 3, ['component_name']] = stand_label
    train_df.loc[train_df['class'] == 2, ['component_name']] = shift_label
    train_df.loc[train_df['class'] == 5, ['component_name']] = short_label
    train_df.loc[train_df['class'] == 4, ['component_name']] = broke_label

    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)

    val_df.loc[val_df['class'] == 1, ['component_name']] = missing_label
    val_df.loc[val_df['class'] == 3, ['component_name']] = stand_label
    val_df.loc[val_df['class'] == 2, ['component_name']] = shift_label
    val_df.loc[val_df['class'] == 5, ['component_name']] = short_label
    val_df.loc[val_df['class'] == 4, ['component_name']] = broke_label

    val_df = pd.concat([val_df, ind_val])
    
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)

    test_df.loc[test_df['class'] == 1, ['component_name']] = missing_label
    test_df.loc[test_df['class'] == 3, ['component_name']] = stand_label
    test_df.loc[test_df['class'] == 2, ['component_name']] = shift_label
    test_df.loc[test_df['class'] == 5, ['component_name']] = short_label
    test_df.loc[test_df['class'] == 4, ['component_name']] = broke_label

    
    test_df = pd.concat([test_df, ind_test])
    
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
#     test_df_mapping2_label = test_df.copy()
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
#     test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

#     name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
#     num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
#     test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

#     for name in set(test_df_mapping2_label['component_name'].values):
#         temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
#         for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
#             if k == 0:
#                 test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
#             elif k ==1:
#                 try:
#                     test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
#                 except:
#                     print(f"{name} only contains bad label.")
#                     test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
#     test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
#     test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
#     test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
#     col = {'overkill': 0, 'leakage': 0}
#     test_component_name_df = test_component_name_df.assign(**col)

#     test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    print('add_test:',add_test)
    if add_test == True:
        # 取得new component的good sample給component classifier訓練
        for name in valComponent:
            good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
            val_df = val_df.drop(good_new_component.index)
            bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
            val_df = val_df.drop(bad_new_component_sample.index)
            train_df = pd.concat([train_df, good_new_component])
        for name in testComponent:
            good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
            test_df = test_df.drop(good_new_component.index)
            train_df = pd.concat([train_df, good_new_component])


#     val_com_df = val_df.copy()
#     good_val_samples = val_com_df.loc[val_com_df['class']==0]
#     missing_val_samples = val_com_df.loc[(val_com_df['component_name']==missing_label)]
#     stand_val_samples = val_com_df.loc[(val_com_df['component_name']==stand_label)]
#     val_com_df = pd.concat([good_val_samples, missing_val_samples, stand_val_samples])


#     train_com_df = train_df.copy()
#     good_samples = train_com_df.loc[train_com_df['class']==0]
#     missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
#     stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
#     train_com_df = pd.concat([good_samples, missing_samples, stand_samples])

    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_com_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==missing_label)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==stand_label)]
    shift_samples = train_com_df.loc[(train_com_df['component_name']==shift_label)]
    short_samples = train_com_df.loc[(train_com_df['component_name']==short_label)]
    broke_samples = train_com_df.loc[(train_com_df['component_name']==broke_label)]
    
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples, shift_samples, short_samples, broke_samples])
    aaa = Counter(train_com_df['component_name'])
    for i in range(max(aaa)):
        if aaa[i] >10000:
            component = train_com_df.loc[train_com_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_com_df[train_com_df['component_name']==i].index
            train_com_df=train_com_df.drop(df_idx)
            train_com_df = pd.concat([train_com_df, component])
    
    
    ### train_defect_df
    train_good_df = train_df.copy()
    train_good_df = train_good_df.loc[train_good_df['class']==0]
    aaa = Counter(train_good_df['component_name'])
    for i in range(max(aaa)):
        if aaa[i] >10000:
            component = train_good_df.loc[train_good_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_good_df[train_good_df['component_name']==i].index
            train_good_df=train_good_df.drop(df_idx)
            train_good_df = pd.concat([train_good_df, component])
            
    train_bad_df = train_df.copy()
    train_bad_df = train_bad_df.loc[train_bad_df['class']!=0]
    aaa = Counter(train_bad_df['component_name'])
    for i in range(max(aaa)):
        if aaa[i] >10000:
            component = train_bad_df.loc[train_bad_df['component_name']==i]
            component = component.sample(n=10000,random_state=123,axis=0)
            df_idx = train_bad_df[train_bad_df['component_name']==i].index
            train_bad_df=train_bad_df.drop(df_idx)
            train_bad_df = pd.concat([train_bad_df, component])
            

            
    train_df = pd.concat([train_good_df, train_bad_df])
    
    ### val_df
    val_good_df = val_df.copy()
    val_good_df = val_good_df.loc[val_good_df['class']==0]
    aaa = Counter(val_good_df['component_name'])
    for i in range(max(aaa)):
        if aaa[i] >5000:
            component = val_good_df.loc[val_good_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_good_df[val_good_df['component_name']==i].index
            val_good_df=val_good_df.drop(df_idx)
            val_good_df = pd.concat([val_good_df, component])
            
    val_bad_df = val_df.copy()
    val_bad_df = val_bad_df.loc[val_bad_df['class']!=0]
    aaa = Counter(val_bad_df['component_name'])
    for i in range(max(aaa)):
        if aaa[i] >5000:
            component = val_bad_df.loc[val_bad_df['component_name']==i]
            component = component.sample(n=5000,random_state=123,axis=0)
            df_idx = val_bad_df[val_bad_df['component_name']==i].index
            val_bad_df=val_bad_df.drop(df_idx)
            val_bad_df = pd.concat([val_bad_df, component])

            
    val_df = pd.concat([val_good_df, val_bad_df])
    print("Num of Images in Component Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df
def CreateTSNEdataset_regroup_fruit(seed , tsne=False):
    
    random.seed(seed)
    add_test=True
#     train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_cls_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_EXP1_csv.csv')
    train_com_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_EXP2_csv.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_test_csv.csv')

    

# 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_regroup_df['component_name'].value_counts().index.tolist()

    if seed == 1: 
        badComponent = [0]
    elif seed == 42: 
        badComponent = [1]
    elif seed == 1212: 
        badComponent = [2]
    else :
        badComponent = random.sample(trainComponent, 1)

    trainDatasetMask = (train_cls_regroup_df['component_name'].isin(badComponent)) & (train_cls_regroup_df['class'] == 1)

#     valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)

    train_cls_regroup_df = train_cls_regroup_df[-trainDatasetMask].copy()
    
    train_cls_regroup_df['component_full_name'] = train_cls_regroup_df['component_name']
    train_com_regroup_df['component_full_name'] = train_com_regroup_df['component_name']
#     val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----

#     train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_cls_regroup_df, transform=val_transform)
    train_com_dataset = TsneCustomDataset(train_com_regroup_df, transform=val_transform)
#     val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     validation_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    train_com_loader = torch.utils.data.DataLoader(
        train_com_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    return training_loader, train_com_loader, test_loader

def CreateTSNEdataset_regroup_fruit_8(seed , tsne=False):
    random.seed(seed)
    add_test=True
#     train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_cls_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_EXP1.csv')
    train_com_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_EXP2.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/fruit_8_csv/fruit_dataset_test.csv')


# 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_regroup_df['component_name'].value_counts().index.tolist()

    badComponent = random.sample(trainComponent, 2)

    trainDatasetMask = (train_cls_regroup_df['component_name'].isin(badComponent)) & (train_cls_regroup_df['class'] == 1)

#     valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)

    train_cls_regroup_df = train_cls_regroup_df[-trainDatasetMask].copy()
#     val_df = val_df[-valDatasetMask].copy()
    
    train_cls_regroup_df['component_full_name'] = train_cls_regroup_df['component_name']
    train_com_regroup_df['component_full_name'] = train_com_regroup_df['component_name']
#     val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----

#     train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=10, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=30, random_state=123))
#     import pdb;pdb.set_trace()
#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_cls_regroup_df, transform=val_transform)
    train_com_dataset = TsneCustomDataset(train_com_regroup_df, transform=val_transform)
#     val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     validation_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    train_com_loader = torch.utils.data.DataLoader(
        train_com_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    return training_loader, train_com_loader, test_loader

def CreateTSNEdataset_mvtecad(seed , tsne=False):
    
    add_test=True
#     train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset_regroup_due_2(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_cls_regroup_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_dataset_csv/mvtecad_dataset_EXP1.csv')
    train_com_regroup_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_dataset_csv/mvtecad_dataset_EXP2.csv')
    test_df = pd.read_csv(f'/root/notebooks/dataset/mvtecad_dataset_csv/mvtecad_dataset_test.csv')
    
    train_cls_regroup_df['component_full_name'] = train_cls_regroup_df['component_name']
    train_com_regroup_df['component_full_name'] = train_com_regroup_df['component_name']
#     val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----

#     train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

#     val_df = pd.concat([val_df_good, val_df_bad])

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)
    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=5, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(n=5, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

#     test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.005, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_cls_regroup_df, transform=val_transform)
    train_com_dataset = TsneCustomDataset(train_com_regroup_df, transform=val_transform)
#     val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     validation_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    train_com_loader = torch.utils.data.DataLoader(
        train_com_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    return training_loader, train_com_loader, test_loader

def CreateTSNEdataset(seed , tsne=False):
    
    add_test=True
    train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df = CreateDataset(seed, add_test)
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    # ---- tsne use less data ----
    
    train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
        
    train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

    val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    val_df = pd.concat([val_df_good, val_df_bad])


    test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

    test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader

def CreateTSNEdatasetRelabel(seed , tsne=False):
       # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成2個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5 
    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()

    newComponent = [4,8,9,12,13,14,20,2,3,5,10,11,17]
    
    # if seed == 11:
    #     valComponent = [4,8,9,12,13,14,20]
    #     testComponent = [2,3,5,10,11,17]

    # for i in valComponent:
    #     trainComponent.remove(i)
    # for i in testComponent:
    #     trainComponent.remove(i)
    # trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(newComponent, 6)
    for i in valComponent:
        newComponent.remove(i)
    testComponent = random.sample(newComponent, 7)
    for i in testComponent:
        newComponent.remove(i)

    if seed == 11:
        valComponent = [4,8,9,12,13,14,20]
        testComponent = [2,3,5,10,11,17]

    for i in valComponent:
        trainComponent.remove(i)
    for i in testComponent:
        trainComponent.remove(i)
    # trainComponent.remove(1) # 元件A (樣本最多的)
    # valComponent = random.sample(trainComponent, 6)
    # for i in valComponent:
    #     trainComponent.remove(i)
    # testComponent = random.sample(trainComponent, 6)
    # for i in testComponent:
    #     trainComponent.remove(i)
    # trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
#     import pdb;pdb.set_trace()
    # Set missing, stand samples as independent components
    train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
    train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
#     train_df.loc[train_df['class'] == 1, 'class'] = 1
#     train_df.loc[train_df['class'] == 2, 'class'] = 1
#     train_df.loc[train_df['class'] == 3, 'class'] = 1
#     train_df.loc[train_df['class'] == 4, 'class'] = 1
#     train_df.loc[train_df['class'] == 5, 'class'] = 1
#     import pdb;pdb.set_trace()
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Val component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    # Set missing, stand samples as independent components
    val_df.loc[val_df['class'] == 1, ['component_name']] = 21
    val_df.loc[val_df['class'] == 3, ['component_name']] = 22
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
#     val_df.loc[val_df['class'] == 1, 'class'] = 1
#     val_df.loc[val_df['class'] == 2, 'class'] = 1
#     val_df.loc[val_df['class'] == 3, 'class'] = 1
#     val_df.loc[val_df['class'] == 4, 'class'] = 1
#     val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Test component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
    test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
#     test_df.loc[test_df['class'] == 1, 'class'] = 1
#     test_df.loc[test_df['class'] == 2, 'class'] = 1
#     test_df.loc[test_df['class'] == 3, 'class'] = 1
#     test_df.loc[test_df['class'] == 4, 'class'] = 1
#     test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label

#     for name in valComponent:
#         good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
#         val_df = val_df.drop(good_new_component.index)
#         train_df = pd.concat([train_df, good_new_component])
#     for name in testComponent:
#         good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
#         test_df = test_df.drop(good_new_component.index)
#         train_df = pd.concat([train_df, good_new_component])

    for name in valComponent:
        good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
        val_df = val_df.drop(good_new_component.index)
        bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
        val_df = val_df.drop(bad_new_component_sample.index)
        train_df = pd.concat([train_df, good_new_component])
    for name in testComponent:
        good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
        test_df = test_df.drop(good_new_component.index)
        train_df = pd.concat([train_df, good_new_component])
    
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    

    # ---- tsne use less data ----
    if tsne == True: 
        train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
        
        train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

        val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        val_df = pd.concat([val_df_good, val_df_bad])


        test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader

def CreateTSNEdatasetRelabel_sixcls(seed , tsne=False):
       # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成2個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5 
    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = [1,3,5,6,10,11,16,17]
    trainDefect = [0,2,4,5]
    
    trainComponentDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainComponentDatasetMask].copy()

    trainDefectDatasetMask = train_df['class'].isin(trainDefect)
    train_df = train_df[trainDefectDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, val_df, test_df = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

#     valDatasetMask = df['component_name'].isin(valComponent)
#     val_df = df[valDatasetMask].copy()
#     print("Val component label: ")
#     val_component_label = val_df['component_name'].value_counts().index.tolist()
#     print(val_component_label)
#     val_component_name=[]
#     print("Val component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in val_component_label:  
#             val_component_name.append(v)
#     print(val_component_name)
#     # Set missing, stand samples as independent components
#     val_df.loc[val_df['class'] == 1, ['component_name']] = 21
#     val_df.loc[val_df['class'] == 3, ['component_name']] = 22
#     # 分成2個class (Good and Bad)
#     val_df.loc[val_df['class'] == 0, 'class'] = 0
# #     val_df.loc[val_df['class'] == 1, 'class'] = 1
# #     val_df.loc[val_df['class'] == 2, 'class'] = 1
# #     val_df.loc[val_df['class'] == 3, 'class'] = 1
# #     val_df.loc[val_df['class'] == 4, 'class'] = 1
# #     val_df.loc[val_df['class'] == 5, 'class'] = 1
#     val_df = pd.concat([val_df, ind_val])
#     testDatasetMask = df['component_name'].isin(testComponent)
#     test_df = df[testDatasetMask].copy()
#     print("Test component label: ")
#     test_component_label = test_df['component_name'].value_counts().index.tolist()
#     print(test_component_label)
#     test_component_name=[]
#     print("Test component name: ")
#     for idx, (k, v) in enumerate(component_dict.items()):
#         if k in test_component_label:  
#             test_component_name.append(v)
#     print(test_component_name)
    
#     test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
#     test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
#     test_df.loc[test_df['class'] == 0, 'class'] = 0
# #     test_df.loc[test_df['class'] == 1, 'class'] = 1
# #     test_df.loc[test_df['class'] == 2, 'class'] = 1
# #     test_df.loc[test_df['class'] == 3, 'class'] = 1
# #     test_df.loc[test_df['class'] == 4, 'class'] = 1
# #     test_df.loc[test_df['class'] == 5, 'class'] = 1
#     test_df = pd.concat([test_df, ind_test])
    
#     with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
#         f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
#                 'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
#                 'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
#                )
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    # test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

    name_of_each_component = test_df_mapping2_label['component_name'].value_counts().index.tolist()
    num_of_image_in_each_component = test_df_mapping2_label['component_name'].value_counts().values
    test_component_name_df = pd.DataFrame(list(zip(name_of_each_component, num_of_image_in_each_component)), columns =['component_name', 'total'])

    for name in set(test_df_mapping2_label['component_name'].values):
        temp_data = test_df_mapping2_label.loc[(test_df_mapping2_label["component_name"] == name)]
        for k, v in zip(temp_data['class'].value_counts().keys(), temp_data['class'].value_counts()):
            if k == 0:
                test_component_name_df.loc[test_component_name_df['component_name'] == name, 'good'] = temp_data['class'].value_counts().sort_index().values[0]
            elif k != 0:
                try:
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label

#     for name in valComponent:
#         good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
#         val_df = val_df.drop(good_new_component.index)
#         train_df = pd.concat([train_df, good_new_component])
#     for name in testComponent:
#         good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
#         test_df = test_df.drop(good_new_component.index)
#         train_df = pd.concat([train_df, good_new_component])

    # for name in valComponent:
    #     good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
    #     val_df = val_df.drop(good_new_component.index)
    #     bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] != 0)]
    #     val_df = val_df.drop(bad_new_component_sample.index)
    #     train_df = pd.concat([train_df, good_new_component])
    # for name in testComponent:
    #     good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
    #     test_df = test_df.drop(good_new_component.index)
    #     train_df = pd.concat([train_df, good_new_component])
    
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    
    train_com_df = train_df.copy()
    good_samples = train_com_df.loc[train_df['class']==0]
    missing_samples = train_com_df.loc[(train_com_df['component_name']==21)]
    stand_samples = train_com_df.loc[(train_com_df['component_name']==22)]
    train_com_df = pd.concat([good_samples, missing_samples, stand_samples])
    

    # ---- tsne use less data ----
    if tsne == True: 
        train_df = train_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
        
        train_com_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

        val_df_good = val_df.loc[val_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        val_df_bad = val_df.loc[val_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        val_df = pd.concat([val_df_good, val_df_bad])


        test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        test_df_bad = test_df.loc[test_df["class"]!=0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123))

        test_df = pd.concat([test_df_good, test_df_bad])
    # -----------------------------

#     val_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
#     train_df = train_df.sample(frac=0.025, random_state=123)
#     val_df = val_df.sample(frac=0.05, random_state=123)
#     test_df = test_df.sample(frac=0.05, random_state=123)

#     train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))

#     test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)

#     test_df_bad = test_df.loc[test_df["class"]!=0].sample(n=50, random_state=123)

#     test_df = pd.concat([test_df_good, test_df_bad])

    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)

#     train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])


    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

#     train_val_loader = torch.utils.data.DataLoader(
#         train_val_dataset, batch_size=128, shuffle=False,
#         num_workers=8, pin_memory=True)

    return training_loader, validation_loader, test_loader 

def get_features_trained_weight(model, dataloader, embedding_layer, tsne=False):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model.eval()
    model.to(device)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None
    
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    name_list = []
    full_name_list = []
    tsne_label_list = []
    print("Start extracting Feature")
    for i, data in enumerate(tqdm(dataloader)):

        try:
            img, target, file_path, name, full_name = data
        except:
            img, target, file_path, name = data
            
        feat_list = []
        def hook(module, input, output):
            feat_list.append(output.clone().detach())
        
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

#             handle=model.feature_extractor.common_embedding.register_forward_hook(hook)
#             handle=model.feature_extractor.embedding.register_forward_hook(hook)
#             handle=model.feature_extractor.encoder.register_forward_hook(hook)
#             handle=model.register_forward_hook(hook)
            # handle=model.feature_extractor.encoder.register_forward_hook(hook)
            handle=model.encoder.register_forward_hook(hook)

#             handle=model.feature_extractor.bn1.register_forward_hook(hook) # model.feature_extractor.layer3

            output = model.forward(images)
#             feat = torch.flatten(feat_list[0], 1)
            feat = torch.flatten(feat_list[0], 1)
            handle.remove()
        
        current_features = feat.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
            
#     import pdb;pdb.set_trace()
            
    return features, labels, image_paths, name_list, full_name_list, tsne_label_list

def get_features_weight(model, dataloader):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    
    model.eval()
    model.to(device)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None
    
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    name_list = []
    full_name_list = []
    tsne_label_list = []
    print("Start extracting Feature")
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

#             if tsne:
#                 for tsne_lb in tsne_label:
#                     tsne_label_list.append(tsne_lb)
        except:
            pass

        with torch.no_grad():

#             handle=model.feature_extractor.common_embedding.register_forward_hook(hook)
#             handle=model.feature_extractor.embedding.register_forward_hook(hook)
#             handle=model.feature_extractor.encoder.register_forward_hook(hook)

#             handle=model.feature_extractor.bn1.register_forward_hook(hook) # model.feature_extractor.layer3

            output = model.forward(images)

#             feat = torch.flatten(output)
#             feat = torch.flatten(feat_list[0], 1)
#             handle.remove()
#         import pdb;pdb.set_trace()

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
    import pdb;pdb.set_trace()     
            
    return features, labels, image_paths, name_list, full_name_list, tsne_label_list

def returnDatasetList(seed, relabel=False):
    random.seed(seed)
    
    if relabel:
        print('==> Preparing relabel data..')
        df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    else:
        print('==> Preparing original label data..')
        df = pd.DataFrame.from_dict(json.load(open(json_path, "r")), orient="index")
        df['file_path'] = df.index
        df["file_path"] = data_dir + df["file_path"].astype(str)
    
        # 分成6個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 1
    df.loc[df['class'] == "stand", 'class'] = 1
    df.loc[df['class'] == "broke", 'class'] = 1
    df.loc[df['class'] == "short", 'class'] = 1
    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]
        # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    return trainComponent, valComponent, testComponent

def CreateTSNEdatasetRelabel_OneComponent(seed, componentName):
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel_ywl.csv")
    
    # 分成2個class
    df.loc[df['class'] == "good", 'class'] = 0
    df.loc[df['class'] == "missing", 'class'] = 1
    df.loc[df['class'] == "shift", 'class'] = 2
    df.loc[df['class'] == "stand", 'class'] = 3
    df.loc[df['class'] == "broke", 'class'] = 4
    df.loc[df['class'] == "short", 'class'] = 5 
    
    # 移除資料集中的Label Noise   
    unwantedData = pd.read_csv(noisy_label_path, sep=",", header=None)[0].tolist()
    df = df[~df.file_path.isin(unwantedData)]
    
    df['component_name'] = labelencoder.fit_transform(df['component_name'])
    component_name_counter = Counter(labelencoder.inverse_transform(df['component_name']))
    component_name_list = [key for key, _ in component_name_counter.most_common()]
    component_label_list = df['component_name'].value_counts().index.tolist()
    component_dict = {component_label_list[i]: component_name_list[i] for i in range(len(component_label_list))}
    # 將Test set從Training set中移除並重新切割資料集
    trainComponent = df['component_name'].value_counts().index.tolist()
    trainComponent.remove(1) # 元件A (樣本最多的)
    valComponent = random.sample(trainComponent, 6)
    for i in valComponent:
        trainComponent.remove(i)
    testComponent = random.sample(trainComponent, 6)
    for i in testComponent:
        trainComponent.remove(i)
    trainComponent.append(1)
    
    trainDatasetMask = df['component_name'].isin(trainComponent)
    train_df = df[trainDatasetMask].copy()
    
    print("Train component label: ")
    train_component_label = train_df['component_name'].value_counts().index.tolist()
    print(train_component_label)
    train_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in train_component_label:  
            train_component_name.append(v)
    print(train_component_name)
    
    # Set missing, stand samples as independent components
    train_df.loc[train_df['class'] == 1, ['component_name']] = 21 # missing
    train_df.loc[train_df['class'] == 3, ['component_name']] = 22 # stand
    train_df.loc[train_df['class'] == 0, 'class'] = 0
    train_df.loc[train_df['class'] == 1, 'class'] = 1
    train_df.loc[train_df['class'] == 2, 'class'] = 1
    train_df.loc[train_df['class'] == 3, 'class'] = 1
    train_df.loc[train_df['class'] == 4, 'class'] = 1
    train_df.loc[train_df['class'] == 5, 'class'] = 1
    
    # 將一部分的In-distribution old component分出來給val set和test set (ind_val, ind_test)
    train_df, ind_val, ind_test = split_stratified_into_train_val_test(train_df, stratify_colname='component_name', frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=seed)

    valDatasetMask = df['component_name'].isin(valComponent)
    val_df = df[valDatasetMask].copy()
    print("Val component label: ")
    val_component_label = val_df['component_name'].value_counts().index.tolist()
    print(val_component_label)
    val_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in val_component_label:  
            val_component_name.append(v)
    print(val_component_name)
    # Set missing, stand samples as independent components
    val_df.loc[val_df['class'] == 1, ['component_name']] = 21
    val_df.loc[val_df['class'] == 3, ['component_name']] = 22
    # 分成2個class (Good and Bad)
    val_df.loc[val_df['class'] == 0, 'class'] = 0
    val_df.loc[val_df['class'] == 1, 'class'] = 1
    val_df.loc[val_df['class'] == 2, 'class'] = 1
    val_df.loc[val_df['class'] == 3, 'class'] = 1
    val_df.loc[val_df['class'] == 4, 'class'] = 1
    val_df.loc[val_df['class'] == 5, 'class'] = 1
    val_df = pd.concat([val_df, ind_val])
    testDatasetMask = df['component_name'].isin(testComponent)
    test_df = df[testDatasetMask].copy()
    print("Test component label: ")
    test_component_label = test_df['component_name'].value_counts().index.tolist()
    print(test_component_label)
    test_component_name=[]
    print("Train component name: ")
    for idx, (k, v) in enumerate(component_dict.items()):
        if k in test_component_label:  
            test_component_name.append(v)
    print(test_component_name)
    
    test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
    test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
    test_df.loc[test_df['class'] == 0, 'class'] = 0
    test_df.loc[test_df['class'] == 1, 'class'] = 1
    test_df.loc[test_df['class'] == 2, 'class'] = 1
    test_df.loc[test_df['class'] == 3, 'class'] = 1
    test_df.loc[test_df['class'] == 4, 'class'] = 1
    test_df.loc[test_df['class'] == 5, 'class'] = 1
    test_df = pd.concat([test_df, ind_test])
    
    with open(f"split_{seed}_component_name_label_mapping.txt", 'w') as f:
        f.write('Train: \n' + str(train_component_name) + '\n' + str(train_component_label) + '\n' +
                'Val: \n' + str(val_component_name) + '\n' + str(val_component_label) + '\n' + 
                'Test: \n' + str(test_component_name) +'\n' + str(test_component_label)
               )
    print("Class distribution in Training set:")
    print(train_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("\nTotal dataset size: ", 
          sum(train_df['class'].value_counts().tolist()) + sum(val_df['class'].value_counts().tolist()) + sum(test_df['class'].value_counts().tolist()))
    print("Num of Images in Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    # 用來產生overkill和leakage數值的dataframe    
    test_df_mapping2_label = test_df.copy()
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 0, 'class'] = 0
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 1, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 2, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 3, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 4, 'class'] = 1
    test_df_mapping2_label.loc[test_df_mapping2_label['class'] == 5, 'class'] = 1

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
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[1]
                except:
                    print(f"{name} only contains bad label.")
                    test_component_name_df.loc[test_component_name_df['component_name'] == name, 'bad'] = temp_data['class'].value_counts().sort_index().values[0]
    test_component_name_df['good'] = test_component_name_df['good'].fillna(0).astype(int)
    test_component_name_df['bad'] = test_component_name_df['bad'].fillna(0).astype(int)
    test_component_name_df = test_component_name_df[['component_name', 'total', 'good', 'bad']]    
    col = {'overkill': 0, 'leakage': 0}
    test_component_name_df = test_component_name_df.assign(**col)

    test_set_class = sorted(test_df['class'].value_counts().keys().tolist())   #由於每個component的label都不一樣，透過這個方式取得該component下的所有label
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_df['component_full_name'] = train_df['component_name']
    val_df['component_full_name'] = val_df['component_name']
    test_df['component_full_name'] = test_df['component_name']
    
    
    test_df = test_df.loc[test_df["component_name"]==componentName]
    test_df_good = test_df.loc[test_df["class"]==0].sample(frac=0.5, random_state=123)
    test_df_bad = test_df.loc[test_df["class"]==1].sample(frac=0.5, random_state=123)
    
    test_dataset_good = TsneCustomDataset(test_df_good, transform=val_transform)
    test_dataset_bad = TsneCustomDataset(test_df_bad, transform=val_transform)
    
    test_loader_good = torch.utils.data.DataLoader(
        test_dataset_good, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)
    
    test_loader_bad = torch.utils.data.DataLoader(
        test_dataset_bad, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return test_loader_good, test_loader_bad
