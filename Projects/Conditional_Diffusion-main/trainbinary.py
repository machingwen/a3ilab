import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from torchsampler import ImbalancedDatasetSampler

from dataset import CompoisitionDataset

import argparse
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import os

def main(args):
    wandb.init(
        project=args.exp,
        config={
            "learning_rate": args.lr,
            "epochs": args.epoch,
        },
        job_type="training"
    )
    save_root = os.path.join(os.getcwd(), 'checkpoints', args.exp, args.dir)
    os.makedirs(save_root, exist_ok=True)

    accelerator = Accelerator()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    attr, obj = args.target.split(' ')

    attr_trainds = CompoisitionDataset(args.data, args.attr_meta, attr, "train", transform)
    attr_valds = CompoisitionDataset(args.data, args.attr_meta, attr, "val", transform)
    balance_sampler = ImbalancedDatasetSampler(attr_trainds, labels=attr_trainds.labels)
    attr_trainloader = torch.utils.data.DataLoader(attr_trainds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    attr_valloader = torch.utils.data.DataLoader(attr_valds, batch_size=args.batch_size, num_workers=args.num_workers)

    obj_trainds = CompoisitionDataset(args.data, args.obj_meta, obj, "train", transform)
    obj_valds = CompoisitionDataset(args.data, args.obj_meta, obj, "val", transform)
    balance_sampler = ImbalancedDatasetSampler(obj_trainds, labels=obj_trainds.labels)
    obj_trainloader = torch.utils.data.DataLoader(obj_trainds, batch_size=args.batch_size, sampler=balance_sampler, num_workers=args.num_workers)
    obj_valloader = torch.utils.data.DataLoader(obj_valds, batch_size=args.batch_size, num_workers=args.num_workers)

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
    attr_optimizer = torch.optim.Adam(attr_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    obj_optimizer = torch.optim.Adam(obj_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    attr_trainloader, attr_valloader, attr_model, attr_optimizer, criterion = accelerator.prepare(attr_trainloader, attr_valloader, attr_model, attr_optimizer, criterion)
    # training attribute classifier
    best_acc = 0
    
    print(f"Training binary {attr} classifier")
    for epoch in range(1, args.epoch + 1):
        attr_model.train()
        progress_bar = tqdm(attr_trainloader, desc=f'Epoch {epoch}')

        for image, label in progress_bar:
            image = image.to(device)
            label = label.unsqueeze(1).to(device)

            logit = attr_model(image)
            loss = criterion(logit, label)
            accelerator.backward(loss)
            wandb.log({"Attr train loss": loss.item()})

            attr_optimizer.step()
            attr_optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        progress_bar = tqdm(attr_valloader, desc=f'Epoch {epoch}')
        attr_model.eval()
        corrects = 0
        for image, label in progress_bar:
            image = image.to(device)
            label = label.unsqueeze(1).to(device)
            with torch.no_grad():
                logit = attr_model(image)
                loss = criterion(logit, label)
                corrects += torch.sum((logit >= 0.5) == label)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            wandb.log({"Attr val loss": loss.item()})
        acc = corrects.item() / len(attr_valds)
        wandb.log({"Attr val acc": acc})
        print(f"Epoch {epoch} Accuracy: {acc:.4f}")

        # save model
        if acc > best_acc:
            best_acc = acc
            torch.save(attr_model.state_dict(), os.path.join(save_root, "best_attr.pth"))

    best_acc = 0
    obj_trainloader, obj_valloader, obj_model, obj_optimizer, criterion = accelerator.prepare(obj_trainloader, obj_valloader, obj_model, obj_optimizer, criterion)
    print(f"Training binary {obj} classifier")
    for epoch in range(1, args.epoch + 1):
        obj_model.train()
        progress_bar = tqdm(obj_trainloader, desc=f'Epoch {epoch}')

        for image, label in progress_bar:
            image = image.to(device)
            label = label.unsqueeze(1).to(device)

            logit = obj_model(image)
            loss = criterion(logit, label)
            accelerator.backward(loss)
            wandb.log({"Obj train loss": loss.item()})

            obj_optimizer.step()
            obj_optimizer.zero_grad()
        
        progress_bar = tqdm(obj_valloader, desc=f'Epoch {epoch}')
        obj_model.eval()
        corrects = 0
        for image, label in progress_bar:
            image = image.to(device)
            label = label.unsqueeze(1).to(device)
            with torch.no_grad():
                logit = obj_model(image)
                loss = criterion(logit, label)
                corrects += torch.sum((logit >= 0.5) == label)
            wandb.log({"Obj val loss": loss.item()})

        acc = corrects.item() / len(obj_valds)
        wandb.log({"Obj val acc": acc})
        print(f"Epoch {epoch} Accuracy: {acc:.4f}")

        # save model
        if acc > best_acc:
            best_acc = acc
            torch.save(obj_model.state_dict(), os.path.join(save_root, "best_obj.pth"))
        
        





    # training obj classifier
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # General Hyperparameters 
    parser.add_argument('--data', type=str, default='/root/notebooks/nfs/work/dataset/conditional_ut', help='dataset location')
    parser.add_argument('--attr_meta', type=str, default="attr_meta.csv")
    parser.add_argument('--obj_meta', type=str, default="obj_meta.csv")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='total training epochs')
    parser.add_argument('--lr_schedule', type=str, default="cosine", choices=["cosine", "piecewise", "linear", "polynomial"])
    
    # Data hyperparameters
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=64, help='training image size')
    
    # Other hyperparameters
    parser.add_argument('--exp', type=str, default='exp', help='experiment directory name')
    parser.add_argument('--dir', type=str, default='NoMiss', help='model weight directory')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='probability of dropping label when training diffusion model')
    parser.add_argument('--target', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
    
    