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
parser.add_argument('--lr_min', '--learning-rate', default=0.06, type=float, help='initial learning rate')
parser.add_argument('--nb_value', default=10, type=int, help='number of lrvalue to test')
parser.add_argument('--lr_max', default=0.5, type=float, help='width of the array of lr to be tested')
parser.add_argument('--size_crop', default=224, type=int, help='size of the image crops')
parser.add_argument('--aug_plus', action='store_true', help='School a more developped augmentation strategy')
parser.add_argument('--chest_aug', action='store_true', help='School a more developped augmentation strategy for chest x ray')
parser.add_argument('--logspace', action="store_true", help='If true value between lr min and lr max will be evenly spaced in log scale')
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
parser.add_argument('--root_folder', default='./data', type=str, metavar='PATH', help='path to the root folder for dataset')
args = parser.parse_args()  


# lr_values = np.linspace(args.lr_min, args.lr_max, num= 10)
lr_values = np.logspace(args.lr_min, args.lr_max, num= args.nb_value) if args.logspace else np.linspace(args.lr_min, args.lr_max, num= args.nb_value) 

if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

def main():
    print("Config:\t", args)
    moco_dataset = MocoDatasetGenerator(args.root_folder, args=args) # add argument for root folder options
    train_dataset = moco_dataset.get_moco_dataset(args.dataset, train_root=os.path.join(args.root_folder, "train")) # add argument for dataset options
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    memory_loader, test_loader = moco_dataset.get_moco_data_loader(args.dataset, args.batch_size, test_root=os.path.join(args.root_folder, "test"))
    print("\n Train set : ", len(train_loader.dataset),
    "\n Memory or test set : ", len(test_loader.dataset),
    "\n Number of class : ", len(train_loader.dataset.classes),
    "\n Args :", args)
    lr_dict={
        'dataset': [],
        'dataset_path':[],
        'lr':[],
        'last_loss': []
    }
    # lr_results=pd.DataFrame(lr_dict)
    file_name="learning_rate_{}_{}.csv".format(args.dataset, args.root_folder)
    for test_lr in lr_values:
        model = ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
        ).cuda()
        args.lr=test_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=test_lr, weight_decay=args.wd, momentum=0.9)
        epoch_start =1
        moco_train = TrainUtils(model = model, train_loader= train_loader, optimizer= optimizer, args= args, args_dict=vars(args), memory_loader=memory_loader, test_loader=test_loader)
        last_loss=moco_train.train(epoch_start)
        lr_dict['lr'].append(test_lr)
        lr_dict['dataset'].append(args.dataset)
        lr_dict['dataset_path'].append(args.root_folder)
        lr_dict['last_loss'].append(last_loss)
    lr_results=pd.DataFrame.from_dict(lr_dict, orient='index')
    lr_results.to_csv(file_name)

if __name__ == "__main__":
    main()
