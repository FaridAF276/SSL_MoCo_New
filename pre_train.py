from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from resnet_moco import ModelBase
from moco_wrapper import ModelMoCo
from moco_dataset_generator import MocoDatasetGenerator
from train_fun import TrainUtils
import yaml
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# knn test flag
parser.add_argument('--knn', action='store_true', help='option to get knn accuracy')
# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

# datasets
parser.add_argument('--dataset', default='cifar10', type=str, help='name of the dataset')


# root folder
parser.add_argument('--root_folder', default='./data', type=str, metavar='PATH', help='path to the root folder for dataset')

args = parser.parse_args()  

if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    print("Shown picture size : ",inp.size())
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.imsave(fname="check_image.jpg", arr=inp)
    plt.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def main():
    #load dataset
    print("Config:\t", args)
    moco_dataset = MocoDatasetGenerator(args.root_folder) # add argument for root folder options
    train_dataset = moco_dataset.get_moco_dataset(args.dataset, train_root=os.path.join(args.root_folder, "train")) # add argument for dataset options
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    memory_loader, test_loader = moco_dataset.get_moco_data_loader(args.dataset, args.batch_size, test_root=os.path.join(args.root_folder, "test"))
    print("\n Train set : ", len(train_loader.dataset),
    "\n Memory or test set : ", len(test_loader.dataset),
    "\n Number of class : ", len(train_loader.dataset.classes),
    "\n Args :", args)
#     _, (inputs, classes) = next(iter(train_loader))
#     class_names=train_loader.dataset.classes
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out)
#     # Make a grid from batch
#     out = torchvision.utils.make_grid(inputs)
    print("Dataset Summary : ")
    from collections import Counter
    print("Sample per class (Train)",dict(Counter(train_loader.dataset.targets)))
    print("Sample per class (Test/Memory)",dict(Counter(memory_loader.dataset.targets)))
    # create model
    model = ModelMoCo(
    dim=args.moco_dim,
    K=args.moco_k,
    m=args.moco_m,
    T=args.moco_t,
    arch=args.arch,
    bn_splits=args.bn_splits,
    symmetric=args.symmetric,
    ).cuda()
   
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))
    else:
        epoch_start =1
    moco_train = TrainUtils(model = model, train_loader= train_loader, optimizer= optimizer, args= args, args_dict=vars(args), memory_loader=memory_loader, test_loader=test_loader)
    
    if(args.knn):
        moco_train.knn_train(epoch_start)
    else:
        moco_train.train(epoch_start)

if __name__ == "__main__":
    main()






