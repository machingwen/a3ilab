# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils import data
from torchvision import datasets, transforms
import math
import numpy as np
import pandas as pd
import json
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from pytorch_metric_learning import samplers
import sys
sys.path.append('/root/notebooks/clustimage_phison/clustimage/')
from clustimage import Clustimage
labelencoder = LabelEncoder()

data_dir = "/root/notebooks/.datasets_for_ma/"
json_path = "/root/notebooks/Phison/dataset.json"
noisy_label_path = "/root/notebooks/Phison/toRemove.txt"


# data_dir = "/root/notebooks/Phison/.datasets_for_ma/"
# json_path = "/root/notebooks/Phison/dataset.json"
# noisy_label_path = "/root/notebooks/Phison/toRemove.txt"
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
        component_name = row["component_name"]
        file_path = row["file_path"]
        return (image, label, file_path, component_name)


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


def get_CIFAR10(root):
    # https://github.com/talreiss/Mean-Shifted-Anomaly-Detection/blob/main/utils.py#L110-L132
    input_size = 32
    num_classes = 10

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Alternative
    # normalize = transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR100(root):
    input_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset



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


def get_fruit2(root, seed):
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
    val_regroup_df = pd.read_csv(f'/root/notebooks/dataset/fruit_dataset_csv/fruit_dataset_test_csv.csv')
    

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



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset

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
    
    
    
    # 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_df['component_name'].value_counts().index.tolist()

    badComponent = random.sample(trainComponent, 2)

    trainDatasetMask = (train_cls_df['component_name'].isin(badComponent)) & (train_cls_df['class'] == 1)
    
    valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)
    
    train_cls_df = train_cls_df[-trainDatasetMask].copy()
    val_df = val_df[-valDatasetMask].copy()


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



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset

def get_fruit_8_stage2(root, seed,component_name):
    
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
    
    
    
    # 將部分bad 類別 從Training set中移除
    trainComponent = train_cls_df['component_name'].value_counts().index.tolist()

    badComponent = random.sample(trainComponent, 2)

    trainDatasetMask = (train_cls_df['component_name'].isin(badComponent)) & (train_cls_df['class'] == 1)
    
    valDatasetMask = (val_df['component_name'].isin(badComponent)) & (val_df['class'] == 1)
    
    train_cls_df = train_cls_df[-trainDatasetMask].copy()
    val_df = val_df[-valDatasetMask].copy()
    
    train_com_df = train_com_df.loc[(train_com_df['component_name'] == component_name)]
    bad_df = train_cls_df.loc[(train_cls_df['class'] == 1)]
    train_cls_df = pd.concat([train_com_df, bad_df])
    
    val_com_df = val_df.loc[(val_df['component_name'] == component_name)]
    val_df = val_df.drop(val_com_df.index)
    val_bad_df = val_df.loc[(val_df['class'] == 1)]
    val_df = pd.concat([val_com_df, val_bad_df])
    
#     import pdb;pdb.set_trace()

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
    
    per_component_num = 32 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 32 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=32, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=32, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, 
        num_workers=8, pin_memory=True,drop_last=True
    )



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_loader ,train_cls_dataset, train_com_dataset, test_com_dataset

def CreateDataset_for_each_component_gp(seed, train_val_df, component_name):
    
    
    train_val_df = train_val_df.loc[(train_val_df['component_name'] == component_name)]
    print("Class distribution in Test set:")
    print(train_val_df['class'].value_counts())
    print("Num of Images in Test set: ", sum(train_val_df['class'].value_counts().tolist()))
    
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_val_dataset = CustomDataset(train_val_df, transform=val_transform)

    
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_val_loader, train_val_dataset

def get_fruit_8_MSSH(root, seed):
    input_size = 256
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
    
    per_component_num = 32 // len(train_com_dataset.dataframe['component_name'].value_counts().index)
    per_class_num = 32 // len(train_cls_dataset.dataframe['class'].value_counts().index)
    train_com_sampler = samplers.MPerClassSampler(train_com_dataset.dataframe['component_name'], per_component_num, length_before_new_iter=len(train_com_dataset))
    train_cls_sampler = samplers.MPerClassSampler(train_cls_dataset.dataframe['class'], per_class_num, length_before_new_iter=len(train_cls_dataset))

    train_com_loader = torch.utils.data.DataLoader(
            train_com_dataset, batch_size=32, shuffle=(train_com_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_com_sampler)

    train_cls_loader = torch.utils.data.DataLoader(
            train_cls_dataset, batch_size=32, shuffle=(train_cls_sampler is None),
            num_workers=8, pin_memory=True,drop_last=True, sampler=train_cls_sampler)



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset

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



    return input_size ,num_classes ,train_com_loader, train_cls_loader, test_dataset ,train_cls_dataset, train_com_dataset, test_com_dataset

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
    
    test_dataset = CustomDataset(test_df, transform=test_transform)

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



    return input_size ,num_classes ,train_com_loader, train_cls_loader, val_dataset ,train_cls_dataset, train_com_dataset, val_com_dataset

    

all_datasets = {
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "CIFAR100": get_CIFAR100,
    'fruit2':get_fruit2,
    'MVtecAD':get_mvtecad,
    'MVtecAD_texture':get_mvtecad_tex,
    'fruit_8':get_fruit_8,
    'fruit_8_s2':get_fruit_8_stage2
}


def get_dataset(dataset, seed, root="./"):
    return all_datasets[dataset](root ,seed)
def get_dataset_bs(dataset, seed, bs, root="./"):
    return all_datasets[dataset](root ,seed, bs)

def get_dataset_com_name(dataset, seed, com_name, root="./"):
    return all_datasets[dataset](root ,seed, com_name)

def get_dataloaders(dataset, train_batch_size=128, root="./"):
    ds = all_datasets[dataset](root)
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": 8, "pin_memory": True}

    train_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader, input_size, num_classes
