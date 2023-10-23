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
labelencoder = LabelEncoder()

# data_dir = "/home/a3ilab01/Downloads/.datasets_for_ma/"
# json_path = "/home/a3ilab01/Downloads/dataset.json"
# noisy_label_path = "/home/a3ilab01/Downloads/toRemove.txt"
# data_dir = "/root/notebooks/Phison/.datasets_for_ma/"
# json_path = "/root/notebooks/Phison/dataset.json"
# noisy_label_path = "/root/notebooks/Phison/toRemove.txt"
data_dir = "/root/notebooks/.datasets_for_ma/"
json_path = "/root/notebooks/Phison/dataset.json"
noisy_label_path = "/root/notebooks/Phison/toRemove.txt"

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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, mode='train'):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
#         image = self.transform(Image.open((row["file_path"])))
        image = self.transform(Image.open((row["file_path"])).convert('RGB'))
        label = np.asarray(row["class"])
        component_name = row["component_name"]
        file_path = row["file_path"]
        return (image, label, file_path, component_name)

class TsneCustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, tsne=False):
        self.dataframe = dataframe
        self.transform = transform
        self.tsne = tsne
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self.transform(Image.open((row["file_path"])))
        label = np.asarray(row["class"])
        component_name = row["component_name"]
        component_full_name = row['component_full_name']
        file_path = row["file_path"]
        return (image, label, file_path, component_name, component_full_name)

def CreateDataset_relabel(seed, testing=None):
    # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel2.csv")

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
#     df_minority_component = train_df.loc[train_df['component_name']==12]
#     train_df = train_df.append([df_minority_component]*3, ignore_index=False)    
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
    if testing is None:
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
        bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 1)]
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
        
    print("Class distribution in Class Training set:")
    print(train_df['class'].value_counts())
    print("Class distribution in Component Training set:")
    print(train_com_df['class'].value_counts())
    print("\nClass distribution in Val set:")
    print(val_df['class'].value_counts())
    print("\nClass distribution in Testing set:")
    print(test_df['class'].value_counts())
    print("Num of Images in Class Training set: ", sum(train_df['class'].value_counts().tolist()))
    print("Num of Images in Component Training set: ", sum(train_com_df['class'].value_counts().tolist()))
    print("Num of Images in Validation set: ", sum(val_df['class'].value_counts().tolist()))
    print("Num of Images in Testing set: ", sum(test_df['class'].value_counts().tolist()))
    return train_df, val_df, test_df, train_component_label, val_component_label, test_component_label, train_com_df

def CreateDataset_relabel_for_each_component(seed, component_name):
    _, _, _, _, _, _, train_df = CreateDataset_relabel(seed)    
    train_df = train_df.loc[(train_df['component_name'] == component_name)]
    print("Class distribution in Training set:")
    print(train_df['class'].value_counts())
    print("Num of Images in Training set: ", sum(train_df['class'].value_counts().tolist()))
    return train_df

def CreateDataset_relabel_for_val_component(seed, component_name):
    _, df, _, _, _, _, _ = CreateDataset_relabel(seed, testing=True) 
    df = df.loc[(df['component_name'] == component_name)]
    return df

def CreateDataset_relabel_for_test_component(seed, component_name):
    _, _, df, _, _, _, _ = CreateDataset_relabel(seed, testing=True)
    df = df.loc[(df['component_name'] == component_name)]
    return df


def CreateTSNEdatasetRelabel(seed, gmm=False):
       # 1:A, 2:B, 3:C, 4:D, 7:F, 8:E
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel2.csv")
    
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
#     df_minority_component = train_df.loc[train_df['component_name']==12]
#     train_df = train_df.append([df_minority_component]*3, ignore_index=False)  
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
    
    if gmm is False:
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
    
    for name in valComponent:
        good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
        val_df = val_df.drop(good_new_component.index)
        train_df = pd.concat([train_df, good_new_component])
        bad_new_component_sample = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 1)]
        val_df = val_df.drop(bad_new_component_sample.index)
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
    
    if gmm:
        train_df = train_com_df.groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.05, random_state=123))
        test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
        test_df_bad = test_df.loc[test_df["class"]==1].sample(n=50, random_state=123)
#         test_df_good = test_df.loc[test_df["class"]==0].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123)) #.sample(n=100, random_state=123)
#         test_df_bad = test_df.loc[test_df["class"]==1].groupby('component_name', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=123)) #.sample(n=100, random_state=123)
        test_df = pd.concat([test_df_good, test_df_bad])
    else:
        train_df = train_df.sample(frac=0.025, random_state=123)
        val_df = val_df.sample(frac=0.05, random_state=123)
        test_df = test_df.sample(frac=0.05, random_state=123)
    
    train_dataset = TsneCustomDataset(train_df, transform=val_transform)
    val_dataset = TsneCustomDataset(val_df, transform=val_transform)
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)
    
    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

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
            if embedding_layer == 'encoder':
                handle=model.encoder.register_forward_hook(hook)
            elif embedding_layer == 'shared_embedding':
                handle=model.embedding.register_forward_hook(hook)
            elif embedding_layer == 'component_classifier':
                handle=model.component_classifier.register_forward_hook(hook)
                
            output = model.forward(images)
            feat = torch.flatten(feat_list[0], 1)
            handle.remove()
        
        current_features = feat.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
            
    return features, labels, image_paths, name_list, full_name_list, tsne_label_list

def CreateTSNEdatasetRelabel_OneComponent(seed, componentName):
    random.seed(seed)
    print('==> Preparing data..')
    df = pd.read_csv("~/Phison/dataset_relabel2.csv")
    
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
#     val_df.loc[val_df['class'] == 1, ['component_name']] = 21
#     val_df.loc[val_df['class'] == 3, ['component_name']] = 22
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

#     test_df.loc[test_df['class'] == 1, ['component_name']] = 21, #'solder_missing'
#     test_df.loc[test_df['class'] == 3, ['component_name']] = 22, #'solder_stand'
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
    
    for name in valComponent:
        good_new_component = val_df.loc[(val_df['component_name'] == name) & (val_df['class'] == 0)].sample(frac=0.5, random_state=123)
        val_df = val_df.drop(good_new_component.index)
        train_df = pd.concat([train_df, good_new_component])
    for name in testComponent:
        good_new_component = test_df.loc[(test_df['component_name'] == name) & (test_df['class'] == 0)].sample(frac=0.5, random_state=123)
        test_df = test_df.drop(good_new_component.index)
        train_df = pd.concat([train_df, good_new_component])
    
    # test_df = test_df.loc[test_df["component_name"]==componentName]
    test_df_good = test_df.loc[test_df["class"]==0].sample(n=50, random_state=123)
    test_df_bad = test_df.loc[test_df["class"]==1].sample(n=50, random_state=123)
    test_df = pd.concat([test_df_good, test_df_bad])
    
    test_dataset = TsneCustomDataset(test_df, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

#     test_loader_good = torch.utils.data.DataLoader(
#         test_dataset_good, batch_size=512, shuffle=False,
#         num_workers=8, pin_memory=True)

#     test_loader_bad = torch.utils.data.DataLoader(
#         test_dataset_bad, batch_size=512, shuffle=False,
#         num_workers=8, pin_memory=True)

    return test_loader
