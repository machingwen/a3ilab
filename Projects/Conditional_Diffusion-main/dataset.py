import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler

from glob import glob
import os
import random
from collections import Counter
import csv


class PointDataset(Dataset):
    def __init__(self, root, transform=None, ignored=None, mean=2.5, std=2.5):
        
        self.root = root
        self.transform = transform
        self.points = np.array([])
        self.context = np.array([])
        self.mean = mean
        self.std = std
        
        for path in glob(os.path.join(self.root, '**', '*.npy')):
            class_label = path.split('/')[-2]
            if class_label == ignored:
                continue
            
            # load points in numpy format
            p = np.load(path)
            p = (p - mean) / std
            n = p.shape[0]
            
            conA, conB = class_label.split(' ')
            conA, conB = int(conA[-1]) - 1, int(conB[-1]) - 1
            cond = np.expand_dims(np.array([conA, conB]), axis=0)
            cond = cond.repeat(n, axis=0)
            
            self.points = np.concatenate((self.points, p), axis=0) if self.points.size else p
            self.context = np.concatenate((self.context, cond), axis=0) if self.context.size else cond
        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.to_list()
        
        points = torch.from_numpy(self.points[index].astype('float32'))
        context = torch.from_numpy(self.context[index].astype('int64'))
        
        return points, context
    

class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None, ignored=None):
        self.root = root
        self.transform = transform
        self.obj_dict = {}
        self.atr_dict = {}
        
        images = glob(os.path.join(root, '**', '*.jpg'))
        self.image_path = []
        self.labels = []
        for image in images:
            cat = image.split('/')[-2]
            if ignored == None or cat not in ignored:
                atr, obj = cat.split(" ")
                self.image_path.append(image)
                self.labels.append((atr, obj))
                
        self.classes = set(self.labels)
                
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image = Image.open(self.image_path[index]).convert('RGB')
        label = self.image_path[index].split('/')[-2]
        label = label.split(' ')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return {"image": image, "atr": self.labels[index][0], "obj": self.labels[index][1]}
    
    def get_class(self):
        return self.obj_dict, self.atr_dict
    
class CustomImageDatasetTripleCond(Dataset):
    def __init__(self, root, transform=None, ignored=None):
        self.root = root
        self.transform = transform
        images = glob(os.path.join(root, '**', '*.jpg'))
        self.image_path = []
        self.labels = []
        for image in images:
            cat = image.split('/')[-2]
            if ignored == None or cat not in ignored:
                size, atr, obj = cat.split(" ")
                self.image_path.append(image)
                self.labels.append((size, atr, obj))
                
        self.classes = set(self.labels)               
                
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image = Image.open(self.image_path[index]).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return {"image": image, "size": self.labels[index][0], "atr": self.labels[index][1], "obj": self.labels[index][2]}
    
class CompoisitionDataset(Dataset):
    def __init__(self, data_root, metadata, target, phase, transform=None):
        self.data, train_data, val_data = [], [], []
        self.labels, label_train, label_val = [], [], []
        self.transform = transform
        self.phase = phase
        if phase == "train" or phase == "val":
            with open(os.path.join(data_root, metadata), newline='') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    img, category, settype = row
                    category = int(category == target)
                    if settype == "train":
                        train_data.append(img)
                        label_train.append(category)
                    elif settype == "val":
                        val_data.append(img)
                        label_val.append(category)
        elif phase == "test":
            self.data = glob(os.path.join(data_root, "*.jpg"))


        if phase == "train":
            self.data = train_data
            self.labels = label_train
        elif phase == "val":
            self.data = val_data
            self.labels = label_val


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = Image.open(self.data[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.phase in ["train", "val"]:
            label = torch.tensor(self.labels[index], dtype=torch.float)
            return img, label
        else:
            return img

class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data
        
    def __iter__(self):
        indices = []
        index = []
        rows = []
        for label in self.data.classes:
            for i in range(len(self.data.labels)):
                if self.data.labels[i] == label:
                    index.append(i)
            indices.append(index)
            index = []
        
        for i in zip(*indices):
            rows.extend(i)
        
        return iter(rows)
    
    def __len__(self):
        return len(self.data)

class ExtendSampler(Sampler):
    def __init__(self, data):
        self.data = data
        self.length = len(data)
        
    def __iter__(self):
        indices = []
        index = []
        rows = []
        max_class_count = max(Counter(self.data.labels).values())
        
        for label in self.data.classes:
            for i in range(len(self.data.labels)):
                if self.data.labels[i] == label:
                    index.append(i)
            num_samples = len(index)
            while num_samples < max_class_count:
                index.extend(random.sample(index, min(max_class_count - num_samples, num_samples)))
                num_samples = len(index)
            indices.append(index[:max_class_count])
            index = []
        
        for i in zip(*indices):
            rows.extend(i)
        self.length = len(rows)

        return iter(rows)

    def __len__(self):
        return self.length

class DecreaseSampler(Sampler):
    def __init__(self, data):
        self.data = data
        self.length = len(data)
        
    def __iter__(self):
        indices = []
        index = []
        rows = []
        min_class_count = min(Counter(self.data.labels).values())
        
        for label in self.data.classes:
            for i in range(len(self.data.labels)):
                if self.data.labels[i] == label:
                    index.append(i)
            indices.append(random.sample(index, min_class_count))
            index = []
        
        for i in zip(*indices):
            rows.extend(i)
        self.length = len(rows)

        return iter(rows)

    def __len__(self):
        return self.length
    