from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))        

        # Affordance.
        self.attr_affordance = {} # -> contains objects compatible with an attribute.
        for _attr in self.attrs:
            candidates = [
                obj
                for (_, attr, obj) in self.train_data
                if attr == _attr
            ]
            self.attr_affordance[_attr] = sorted(list(set(candidates)))
            if len(self.attr_affordance[_attr]) <= 1:
                print(f'{_attr} is associated with <= 1 object: {self.attr_affordance[_attr]}')

        # Images that contain an object.
        self.image_with_obj = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
            self.image_with_obj[obj].append(i)
        
        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)  
        


        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)
        
        
        unseen_pairs = set()
        for pair in self.val_pairs + self.test_pairs:
            if pair not in self.train_pair2idx:
                unseen_pairs.add(pair)
        self.unseen_pairs = list(unseen_pairs)
        self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    

    def __len__(self):
        return len(self.data)
    
    def sample_same_attribute(self, attr, obj, with_different_obj=True):
        if with_different_obj:
            if len(self.attr_affordance[attr]) == 1:
                return -1
            i2 = np.random.choice(self.image_with_attr[attr])
            img1, attr1, obj1 = self.data[i2]
            while obj1 == obj:
                i2 = np.random.choice(self.image_with_attr[attr])
                img1, attr1, obj1 = self.data[i2]
            assert obj1 != obj
        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return i2


    def sample_same_object(self, attr, obj, with_different_attr=True):
        i2 = np.random.choice(self.image_with_obj[obj])
        if with_different_attr:
            img1, attr1, obj1 = self.data[i2]
            while attr1 == attr:
                i2 = np.random.choice(self.image_with_obj[obj])
                img1, attr1, obj1 = self.data[i2]
        return i2

    
    def __getitem__(self, index):
        image, attr, obj = self.data[index]

        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.train_pair2idx[(attr, obj)],
                'img_name': self.data[index][0]
            }

            data['mask_task'] = 1 # Attribute task
            i2 = self.sample_same_attribute(attr, obj, with_different_obj=True)  #隨機找一張同樣att的圖片
            if i2 == -1:
                data['mask_task'] = 0
            img1, attr1, obj1_a = self.data[i2]


            img1 = self.loader(img1)
            img1 = self.transform(img1)

            data['img1_a'] = img1
            data['attr1_a'] = self.attr2idx[attr1]
            data['obj1_a'] = self.obj2idx[obj1_a]
            data['idx1_a'] = i2
            data['img1_name_a'] = self.data[i2][0]

            # Object task.
            i2 = self.sample_same_object(attr, obj, with_different_attr=True)
            img1, attr1_o, obj1 = self.data[i2]


            img1 = self.loader(img1)
            img1 = self.transform(img1)
            data['img1_o'] = img1
            data['attr1_o'] = self.attr2idx[attr1_o]
            data['obj1_o'] = self.obj2idx[obj1]
            data['idx1_o'] = i2
            data['img1_name_o'] = self.data[i2][0]


            if (attr1_o, obj1_a) in self.unseen_pair2idx:
                data['composed_unseen_pair'] = self.unseen_pair2idx[(attr1_o, obj1_a)]
                data['composed_seen_pair'] = 2000
            elif (attr1_o, obj1_a) in self.train_pair2idx:
                data['composed_seen_pair'] = self.train_pair2idx[(attr1_o, obj1_a)]
                data['composed_unseen_pair'] = 2000
            else:
                data['composed_unseen_pair'] = 2000
                data['composed_seen_pair'] = 2000

        else:
            # Testing mode.
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
            }
        return data
