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