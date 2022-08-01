from sched import scheduler
import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from  train_fun import TrainUtils
import logging
from earlystop import EarlyStopping
from moco_wrapper import ModelMoCo
from moco_dataset_generator import FolderPair
import math
import sys
parser = argparse.ArgumentParser(description='PyTorch MoCo Linear Eval')
parser.add_argument('-pt-ssl','--pre-train-ssl', action='store_true', \
    help='What backend to use for pretraining. Boolean. \
        Default pt_ssl=False, i.e, ImageNet pretrained network. ')
parser.add_argument('--model-dir', default='', type=str, metavar='PATH', help='path to directory where pretrained model is saved')
parser.add_argument('--epochs', '-e', default=100, type=int, metavar='N', help='number of epochs')
parser.add_argument('--patience', default=7, type=int, help='patience for the early stopping')
parser.add_argument('--dataset-ft', type=str, help='name of the dataset to fine tune the pretrained model on')
parser.add_argument('--results_dir', type=str, help='name of the path to save the fine tuned model on')
parser.add_argument('--batch_size',default=256, type=int, help='Number of images in the each batch')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--root_folder',default='', type=str, help='folder where dataset is, it has to have train and test folder in it')
parser.add_argument('--num_classes',default=10, type=int, help='Amount of classes in the dataset')
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

args = parser.parse_args()
def cosine_lr(epoch, args):
    lr=args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    return lr
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_stl10_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.STL10('./data', split='train', download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10('./data', split='test', download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=args.batch_size):
    train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10('./data', train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader
def get_folder_data_loaders(shuffle=True, batch_size=args.batch_size, root_folder=''):
    folder_transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_root, test_root = os.path.join(root_folder, "train"), os.path.join(root_folder, "test")
    train_dataset = datasets.ImageFolder(root=train_root, transform=folder_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, shuffle=shuffle, pin_memory=True)

    test_dataset = datasets.ImageFolder(root= test_root, transform=folder_transform)

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, shuffle=shuffle, pin_memory=True)
    return train_loader, test_loader

def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #update args.device
    print("Using device:", device) 
    logging.basicConfig(filename=os.path.join(args.model_dir, 'linear_eval.log'), level=logging.INFO)
           # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    if not args.pre_train_ssl: #use ImageNet pretrained network
        pass
    else: # use SSL pre_trained network
        with open(os.path.join(args.model_dir ,'config.yml'), 'r') as file: #should be run_dir+config.yml. run_dir is user input
            config = yaml.load(file, Loader=yaml.FullLoader)
        if config['arch'] == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).cuda()
        elif config['arch'] == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes).cuda()
        checkpoint = torch.load(os.path.join(args.model_dir, 'model.pth'))
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
        
        if(args.dataset_ft):
            logging.info(f'Fine tune on Dataset: {args.dataset_ft}')
            #print(f'Fine tune on Dataset: {args.dataset_ft}')
            if(args.dataset_ft == 'cifar10'):
                train_loader, test_loader = get_cifar10_data_loaders(download=True)
            elif(args.dataset_ft =='stl10'):
                train_loader, test_loader = get_stl10_data_loaders(download=True)
            elif(args.dataset_ft =='folder'):
                train_loader, test_loader = get_folder_data_loaders(batch_size=args.batch_size, root_folder=args.root_folder)
        else:
            logging.info(f'Fine tune on Dataset: {config["dataset"]}')
            print(f'Fine tune on Dataset: {config["dataset"]}')
            if config['dataset'] == 'cifar10':
                train_loader, test_loader = get_cifar10_data_loaders(download=True)
            elif config['dataset'] == 'stl10':
                train_loader, test_loader = get_stl10_data_loaders(download=True)
            elif(config['dataset'] =='folder'):
                train_loader, test_loader = get_folder_data_loaders(batch_size=args.batch_size, root_folder=args.root_folder)

        print("\n Train set : ", len(train_loader.dataset),
              "\n Memory or test set : ", len(test_loader.dataset),
               "\n Number of class : ", len(train_loader.dataset.classes),
               "\n Args :", args)
            #    "\n Labels :", train_loader.dataset.class_to_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if (not args.cos):
        schedul= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.7, patience=2, threshold=8e-2, verbose=True)
        # schedul=torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=15, gamma=0.1, verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss().cuda()
#     train_ut=TrainUtils(model = model, train_loader= train_loader, optimizer= optimizer, args= args, args_dict=vars(args), memory_loader=test_loader, test_loader=test_loader)

    Early_stop=EarlyStopping(patience=args.patience, verbose=True, path="model_fine.pth")
    epochs = args.epochs
    for epoch in range(1, epochs+1):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            # print(y_batch.size())
            logits = model(x_batch)
            # print(y_batch.size())
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
#             train_ut.adjust_learning_rate(optimizer=optimizer, epoch=epoch, args=args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cosine_lr(epoch, args) if args.cos else schedul.step(loss)
        # schedul.print_lr()
        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top3_accuracy = 0
        Early_stop(loss.item(), model, optimizer, args, epoch)
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            logits = model(x_batch)

            top1, top3 = accuracy(logits, y_batch, topk=(1,3))
            top1_accuracy += top1[0]
            top3_accuracy += top3[0]
        
        top1_accuracy /= (counter + 1)
        top3_accuracy /= (counter + 1)
        if Early_stop.early_stop:
                print("Model not improving, stopping training")
                break;
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, os.path.join(args.results_dir,'model_fine.pth'))
        logging.info("Loss {:.2f}\t Epoch {}\tTrain Acc@1 {:.2f}\tTest Acc@1: {:.2f}\tTest Acc@3: {:.2f}".format(loss,epoch,top1_train_accuracy.item(),top1_accuracy.item(),top3_accuracy.item()))
        print("Lr:{}\t Loss {:.2f}\t Epoch {}\tTrain Acc@1 {:.2f}\tTest Acc@1: {:.2f}\tTest Acc@3: {:.2f}".format(args.lr, loss,epoch,top1_train_accuracy.item(),top1_accuracy.item(),top3_accuracy.item()))


if __name__ == "__main__":
    main()
