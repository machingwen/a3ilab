# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns

import argparse
import os
from multiprocessing import cpu_count
import time
from pprint import pformat
import logging
import sys
from typing import Dict, Any, List, Tuple
from networks.mobilenetv3_HybridExpert import SupConMobileNetV3Large
from util_mo import *
import torch.backends.cudnn as cudnn
from due import dkl_Phison_mo
from lib.datasets_mo import get_dataset
from gpytorch.likelihoods import SoftmaxLikelihood
import gpytorch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['figure.dpi'] = 150

def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_plot_coordinates(image,x,y,image_centers_area_size,offset):

    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    xmin = center_x - int(image_width / 2)
    ymin = center_y - int(image_height / 2)
    xmax = xmin + image_width
    ymax = ymin + image_height

    return xmin, ymin, xmax, ymax    

def plot_scatter_all(args,args_due, X, y, X_test, y_test):    
    random.seed(1)
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx

    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255   
        if args_due.test_tsne == True:
            if idx == 0:
                ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
            else:
                ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component


    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    if args_due.test_tsne == True:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train+test_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    else:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_component_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)

def plot_scatter(args,args_due, X, y, X_test, y_test):    
    tx = X[:, 0]
    ty = X[:, 1]
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")

#     colors_per_class = {}
#     colors_per_class[0] = [0, 0, 255]
#     colors_per_class[1] = [255, 0, 0]
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx
#     import pdb;pdb.set_trace()
    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
        
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255    
        if args_due.test_tsne == True:
            if idx == 0:
                ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
            else:
                ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component
    
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    if args_due.test_tsne == True:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train+test_component_HBE+_class_tsne.pdf".format(args_due.output_inference_dir, args_due.output_inference_dir,args["embedding_layer"]))
    else:
        tnse_points_path= os.path.join(args["output_dir"], "{}/{}_{}_train_component_HBE+_class_tsne.pdf".format(args_due.output_inference_dir,args_due.output_inference_dir, args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)




def set_model(args, args_due,train_com_loader,num_com):
    input_size = 224
    num_classes = num_com
    
    if args_due.n_inducing_points is None:
        args_due.n_inducing_points = num_classes
    n_inducing_points = args_due.n_inducing_points

    if args_due.coeff == 1:
        from networks.mobilenetv3_SN1 import SupConMobileNetV3Large
    elif args_due.coeff == 3:
        from networks.mobilenetv3_SN3 import SupConMobileNetV3Large
    elif args_due.coeff == 5:
        from networks.mobilenetv3_SN5 import SupConMobileNetV3Large
    elif args_due.coeff == 7:
        from networks.mobilenetv3_SN7 import SupConMobileNetV3Large
    elif args_due.coeff == 0:
        from networks.mobilenetv3 import SupConMobileNetV3Large
    
    feature_extractor =  SupConMobileNetV3Large()

    initial_inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values(
            train_com_loader, feature_extractor, n_inducing_points*30
        )

    gp = dkl_Phison_mo.GP(
            num_outputs=num_com, #可能=conponent 數量 = 23個 
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args_due.kernel,
    )

    model = dkl_Phison_mo.DKL(feature_extractor, gp)
    likelihood = SoftmaxLikelihood(num_classes=num_com, mixing_weights=False)
    
    
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')


    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        cudnn.benchmark = True
        model.load_state_dict(ckpt)
        
    return model, likelihood



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--embedding_layer",
        type=str,
        default="shared_embedding",
        help="Which embedding to visualization( encoder or shared_embedding)"
    )
    parser.add_argument(
        "--name",
        type=int,
        default=15,
        help="Test component name"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed"
    )
    parser.add_argument(
        "--no_spectral_conv",
        action="store_false",
        dest="spectral_conv",
        help="Don't use spectral normalization on the convolutions",
    )

    parser.add_argument(
        "--no_spectral_bn",
        action="store_false",
        dest="spectral_bn",
        help="Don't use spectral normalization on the batch normalization layers",
    )
    parser.add_argument(
        "--no_test_tsne",
        action="store_false",
        dest="test_tsne",
        help="Don't use testing set on T-sne",
    )
    parser.add_argument(
        "--coeff", type=float, default=1, help="Spectral normalization coefficient"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--n_power_iterations", default=1, type=int, help="Number of power iterations"
    )
    parser.add_argument(
        "--kernel",
        default="RBF",
        choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
        help="Pick a kernel",
    )
    parser.add_argument(
        "--dataset",
        default="fruit_8",
        choices=["CIFAR10", "CIFAR100", "PHISON",'PHISON_regroup','fruit'],
        help="Pick a dataset",
    )
    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )
    parser.add_argument(
        "--n_inducing_points_cls", type=int, help="Number of inducing points"
    )
    parser.add_argument(
        "-oid", "--output_inference_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument('--relabel', action='store_true', help='relabel dataset')
    args: Dict[str, Any] = vars(parser.parse_args())
    args_due = parser.parse_args()
    print('test_tsne:',args_due.test_tsne)
    set_random_seed(args["random_seed"])

    # Create output directory if not exists
    if not os.path.isdir(args["output_dir"]):
        os.makedirs(args["output_dir"])
        logging.info(f"Created output directory {args['output_dir']}")

    # Initialize device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initialized device {device}")

    # Load model's checkpoint
    loc = 'cuda:0'
    
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")
    ds = get_dataset(args_due.dataset ,args_due.random_seed , root="./data")
    input_size ,num_classes , train_com_loader, train_loader, test_dataset ,train_cls_dataset,train_com_dataset, test_com_dataset = ds

    # Intialize model
    model , likelihood = set_model(args, args_due, train_com_loader , num_classes)
    

    feature_extractor = model.feature_extractor.encoder
    initial_inducing_points, initial_lengthscale = dkl_Phison_mo.initial_values2(
        train_com_loader, feature_extractor, args_due.n_inducing_points*30
    )
    
    # Initialize dataset and dataloader

    training_loader, train_com_loader, test_loader = CreateTSNEdataset_regroup_fruit_8(args["random_seed"], tsne=True)


    # Calculate embeddings from images in reference set
    start = time.time()
    embeddings_train, labels_train, _, name_list_train, _, _ = get_features_trained_weight(model, train_com_loader, embedding_layer=args["embedding_layer"], tsne=True)
    embeddings_test, labels_test, _, name_list_test, _, _ = get_features_trained_weight(model, test_loader, embedding_layer=args["embedding_layer"], tsne=True)   

    end = time.time()
    logging.info(f"Calculated {len(embeddings_train)+len(embeddings_test)} embeddings: {end - start} second")


    # Train + Test set

    embeddings = np.concatenate((embeddings_train, embeddings_test), axis=0)
    
    component_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            component_labels_test_mapping.append(len(set(name_list_train)))
        if lb == 1:
            component_labels_test_mapping.append(len(set(name_list_train))+1)
    
    class_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            class_labels_test_mapping.append(len(set(labels_train)))
        if lb == 1:
            class_labels_test_mapping.append(len(set(labels_train))+1)
    
    # Init T-SNE
    tsne = TSNE(n_components=2, random_state=12345, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    X_transformed = tsne.fit_transform(embeddings)    
    
    
    tsne_train = X_transformed[0:len(embeddings_train)]
    tsne_test = X_transformed[len(embeddings_train):(len(embeddings_train)+len(embeddings_test))] 
    tsne_initial_inducing_points = X_transformed[(len(embeddings_train)+len(embeddings_test))::]
    
    plot_scatter_all(args, args_due, tsne_train, name_list_train, tsne_test, component_labels_test_mapping )
    
    labels = labels_train+labels_test
    plot_scatter(args, args_due, tsne_train, labels_train, tsne_test, class_labels_test_mapping)


