import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset import CompoisitionDataset

import argparse
from accelerate import Accelerator
import os
from collections import OrderedDict
from tqdm import tqdm
import csv

def main(args):

    accelerator = Accelerator()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.save_folder = os.path.join(args.save_folder, args.attr_ckpt.split('/')[-2])
    os.makedirs(args.save_folder, exist_ok=True)
    # Load Dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_ds = CompoisitionDataset(args.data, metadata="", target="", phase="test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load model
    attr_model = resnet18(weights=None)
    obj_model = resnet18(weights=None)

    attr_model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
    )

    obj_model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
    )

    attr_ckpt = torch.load(args.attr_ckpt)
    obj_ckpt = torch.load(args.obj_ckpt)

    new_dict = OrderedDict()
    for k, v in attr_ckpt.items():
        if k.startswith("module"):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    try:
        attr_model.load_state_dict(new_dict)
        print("Attribute model all keys successfully match")
    except:
        print("Attribute model some keys are missing!")

    new_dict = OrderedDict()
    for k, v in obj_ckpt.items():
        if k.startswith("module"):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    try:
        obj_model.load_state_dict(new_dict)
        print("Object model all keys successfully match")
    except:
        print("Object model some keys are missing!")

    attr_model, obj_model, test_loader = accelerator.prepare(attr_model, obj_model, test_loader)

    # start testing
    attr_model.eval()
    obj_model.eval()
    i = 0
    attr_corrects = 0
    obj_corrects = 0
    total_corrects = 0
    
    progress_bar = tqdm(test_loader, desc="Testing")
    for image in progress_bar:
        image = image.to(device)
        with torch.no_grad():
            attr_pred = F.sigmoid(attr_model(image))
            obj_pred = F.sigmoid(obj_model(image))

            attr_pred = attr_pred >= 0.5
            obj_pred = obj_pred >= 0.5
            attr_corrects += torch.sum(attr_pred).item()
            obj_corrects += torch.sum(obj_pred).item()
            match = (attr_pred * obj_pred).view(-1)
            total_corrects += torch.sum(match).item()
            for img in image[match]:
                img = img * 0.5 + 0.5
                save_image(img, os.path.join(args.save_folder, f"{i:05d}.jpg"))
                i += 1
                
    with open(os.path.join(args.save_folder, "result.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Attribute corrects", "Object corrects", "All corrects"])
        writer.writerow([attr_corrects, obj_corrects, total_corrects])
        
    print(f"Select a total of {i} images!")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General Hyperparameters 
    parser.add_argument('--data', type=str, default='/root/notebooks/nfs/work/dataset/conditional_ut', help='dataset location')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_folder', type=str, default="Corrects")

    
    # Data hyperparameters
    parser.add_argument("--attr_ckpt", type=str)
    parser.add_argument("--obj_ckpt", type=str)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=64, help='training image size')
    
    # Other hyperparameters
    parser.add_argument('--drop_prob', type=float, default=0.1, help='probability of dropping label when training diffusion model')
    
    args = parser.parse_args()
    main(args)