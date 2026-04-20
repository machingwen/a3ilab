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
import pandas as pd
import glob



class CustomImageDatasetCondition_MPD(Dataset):
    def __init__(self, data_root, dataset_type, transform):
        self.data_root = data_root
        if dataset_type == "train":
            self.gt_path = os.path.join(data_root, "train_crop_80")  
            self.img_path = os.path.join(data_root, "train_original_80")  
        elif dataset_type == "test":
            self.gt_path = os.path.join(data_root, "test_crop_20")  
            self.img_path = os.path.join(data_root, "test_original_20")
        else:
            raise ValueError("dataset_type 必須是 'train' 或 'test'")
        
        self.transform = transform
        self.obj_dict = {}
        
        # 這一行是必需的來正確初始化 gt_path_files
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.PNG"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        # 從 img_path 中取得原始圖片檔案名稱列表
        self.img_path_files = sorted(glob.glob(os.path.join(self.img_path, "*.PNG")))  # 假設副檔名為 .PNG
                    
                
    def __len__(self):
        # 返回數據集大小，即 CSV 檔案中有多少筆資料
        return len(self.gt_path_files)

    def __getitem__(self, index):        
        img_name = os.path.basename(self.gt_path_files[index])
        img = Image.open(os.path.join(self.img_path, img_name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_path, img_name)).convert('RGB') #原來我的模型的
        #gt = Image.open(os.path.join(self.gt_path, img_name)).convert('L') #給segdiff的
        if self.transform is not None:
            image_input = self.transform(img)
            image_groundtruth = self.transform(gt)
        
        return {"input_image": image_input, 
                "groundtruth_image": image_groundtruth,
                "filename": img_name.split('.')[0]
               }

"""
class CustomImageDatasetCondition_my(Dataset):
    def __init__(self, data_root, dataset_type, transform):
        self.data_root = data_root
        if dataset_type == "train":
            self.gt_path = os.path.join(data_root, "train_crop_80")  
            self.img_path = os.path.join(data_root, "train_original_80")  
        elif dataset_type == "test":
            self.gt_path = os.path.join(data_root, "test_crop_20")  
            self.img_path = os.path.join(data_root, "test_original_20")
        else:
            raise ValueError("dataset_type 必須是 'train' 或 'test'")
        
        self.transform = transform
        self.obj_dict = {}
        
        # 這一行是必需的來正確初始化 gt_path_files
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.PNG"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        # 從 img_path 中取得原始圖片檔案名稱列表
        self.img_path_files = sorted(glob.glob(os.path.join(self.img_path, "*.PNG")))  # 假設副檔名為 .PNG
                    
                
    def __len__(self):
        # 返回數據集大小，即 CSV 檔案中有多少筆資料
        return len(self.gt_path_files)

    def __getitem__(self, index):        
        img_name = os.path.basename(self.gt_path_files[index])
        img = Image.open(os.path.join(self.img_path, img_name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_path, img_name)).convert('RGB')
        if self.transform is not None:
            image_input,image_groundtruth = self.transform(img,gt)
            
            #image_groundtruth = self.transform(gt)
        
        return {"input_image": image_input, 
                "groundtruth_image": image_groundtruth,
                "filename": img_name.split('.')[0]
               }
"""
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
    def __init__(self, csv_file, transform=None, ignored=None):
        #self.root = root
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.obj_dict = {}
        self.atr_dict = {}
        
        # 從 CSV 檔案中提取標準圖片 (Filename_Input) 和瑕疵圖片 (Filename_Groundtruth)
        self.image_input_paths = self.data['Filename_Input'].values
        self.image_x0 = self.data['Filename_Groundtruth'].values
        self.defects = self.data['Defect'].values  # 瑕疵類型
        self.image_groundtruth_paths = self.data['Filename_Groundtruth'].values
        
        self.ATR2IDX = {atr: idx for idx, atr in enumerate(set(self.defects))}
        self.OBJ2IDX = {obj: idx for idx, obj in enumerate(set(self.image_input_paths))}
        
        # 將瑕疵類型 (Defect) 設置為標籤 (labels)
        self.labels = self.defects  # 新增這行，將標籤設置為瑕疵類型
        
        # 定義 classes 屬性（唯一的 Defect 標籤）
        self.classes = list(set(self.defects))  # 新增這行，定義唯一的瑕疵類型
        
        # 建立物件和屬性對應字典（可以根據 Defect 來建）
        obj = list(set(self.image_input_paths))  # 所有標準圖片
        atr = list(set(self.defects))  # 所有瑕疵種類
        
         
        for i in range(len(obj)):
            self.obj_dict[obj[i]] = i
        
        for i in range(len(atr)):
            self.atr_dict[atr[i]] = i
        
        if ignored:
            filter_mask = ~self.data['Defect'].isin(ignored)
            self.data = self.data[filter_mask]
        
        print("Total training images:", len(self.data))
                    
                
    def __len__(self):
        # 返回數據集大小，即 CSV 檔案中有多少筆資料
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        # 讀取瑕疵圖片 (Groundtruth) 和標準圖片 (Input)
        image_input = Image.open(self.image_input_paths[index]).convert('RGB')
        image_groundtruth = Image.open(self.image_groundtruth_paths[index]).convert('RGB')
        defect = self.defects[index]  # 瑕疵標籤
        
        
        if self.transform is not None:
            image_input = self.transform(image_input)
            image_groundtruth = self.transform(image_groundtruth)
        
        return {"input_image": image_input, 
                "groundtruth_image": image_groundtruth, 
                "defect": defect}
    
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
        #print("Counter(self.data.labels):",Counter(self.data.labels))
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
