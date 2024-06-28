import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler
from collections import Counter
from glob import glob
import os


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
        
        
        images = glob(os.path.join(root, '**', '*.jpg'))
        self.image_path = []
        self.labels = []
        for image in images:
            cat = image.split('/')[-2]
            if ignored == None or cat not in ignored:
                atr, obj = cat.split(" ")
                self.image_path.append(image)
                self.labels.append((atr, obj))
                #self.labels.append((cat))
                
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
        #return {"image": image, "text": self.labels[index]}
    
    def get_class(self):
        return self.obj_dict, self.atr_dict

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
            
# class CustomSampler(Sampler):
#     def __init__(self, data):
#         self.data = data
        
#     def __iter__(self):
#         indices = []
#         index = []
#         rows = []
#         max_class_count = max(Counter(self.data.labels).values())  # 找到最多的类别的样本数量
        
#         for label in self.data.classes:
#             for i in range(len(self.data.labels)):
#                 if self.data.labels[i] == label:
#                     index.append(i)
#             # 重复采样，直到该类别的样本数量等于最多类别的样本数量
#             num_samples = len(index)
#             while num_samples < max_class_count:
#                 index.extend(random.sample(index, min(max_class_count - num_samples, len(index))))
#                 num_samples = len(index)
#             indices.append(index[:max_class_count])  # 截取指定数量的样本
#             index = []
        
#         for i in zip(*indices):
#             rows.extend(i)
        
#         return iter(rows)
    
#     def __len__(self):
#         max_class_count = max(Counter(self.data.labels).values())  # 找到最多的類別的樣本數量
#         num_classes = len(self.data.classes)  # 類別的數量
#         return max_class_count * num_classes  
    
    
    
def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def generate_circle(center, radius, num_samples) -> np.ndarray:
    """ generate points inside a circle with cetner and radius """
    theta = np.linspace(0, 2*np.pi, num_samples)
    centerX, centerY = center
    a, b = radius * np.cos(theta) + centerX, radius * np.sin(theta) + centerY

    r = np.random.rand((num_samples)) * radius
    x, y = r * np.cos(theta) + centerX, r * np.sin(theta) + centerY
    return np.stack((x, y), axis=1)