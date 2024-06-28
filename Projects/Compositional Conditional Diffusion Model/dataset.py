import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler
import random
from glob import glob
import os
from torch.utils.data import WeightedRandomSampler

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
        obj = []
        atr = []
        labels = [label.split(' ') for label in os.listdir(root)]
        for l in labels:
            atr.append(l[0])
            obj.append(l[1])
            
        
        obj = list(set(obj))
        atr = list(set(atr))
         
        for i in range(len(obj)):
            self.obj_dict[obj[i]] = i
        
        for i in range(len(atr)):
            self.atr_dict[atr[i]] = i
        
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

# class CustomSampler(Sampler):
#     def __init__(self, data):
#         self.data = data
        
#     def __iter__(self):
#         indices = []
#         index = []
#         rows = []
#         for label in self.data.classes:
#             for i in range(len(self.data.labels)):
#                 if self.data.labels[i] == label:
#                     index.append(i)
#             indices.append(index)
#             index = []
        
#         for i in zip(*indices):
#             rows.extend(i)
        
#         return iter(rows)
    
#     def __len__(self):
#         return len(self.data)
class CustomSampler(Sampler):
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
            
    
    
class Phison:
    
    def __init__(self, transform):
        self.cond2idx = {'good': 0, 'shift': 1, 'broke': 2, 'short': 3} 
        self.comp2idx = {'C0201': 0, 'LED0603': 1, 'SOT23': 2, 'TVS523': 3, 'R0805': 4, 'BGA4P': 5}

        self.idx2cond = {v: k for k, v in self.cond2idx.items()}
        self.idx2comp = {v: k for k, v in self.comp2idx.items()}
        
        self.transform = transform
        
        ds = load_dataset("barry556652/special_500", split="train")
        self.image_column, self.caption_column = ds.column_names
        self.ds = ds.with_transform(self.preprocess_train)
    
    def preprocess_train(self, examples):
        labels = [caption.split(" ") for caption in examples[self.caption_column]]
        images = [image.convert("RGB") for image in examples[self.image_column]]
        examples["pixel_values"] = [self.transform(image) for image in images]
        examples["condition_label"] = [torch.tensor(self.cond2idx[c[0]], dtype=torch.int64) for c in labels]
        examples["class_label"] = [torch.tensor(self.comp2idx[c[1]], dtype=torch.int64) for c in labels]
        return examples
    
    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        class_label = torch.stack([example["class_label"] for example in examples])
        condition_label = torch.stack([example["condition_label"] for example in examples])
        return {"pixel_values": pixel_values, "condition_label": condition_label, "class_label": class_label}
    
    def get_loader(self, batch_size=64, num_workers=4):
        loader = DataLoader(self.ds, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=num_workers)
        return loader
    
    def get_idx2cond(self):
        return self.idx2cond
    
    def get_idx2comp(self):
        return self.idx2comp
    
    
    
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