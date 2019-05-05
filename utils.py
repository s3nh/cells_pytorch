
from __future__ import print_function, division 

import torch
import torch.nn as nn 
import torch.optim as optim 

import numpy as np 
import torchvision

from torchvision import datasets, models, transforms 
import matplotlib.pyplot as plt
import time 
import os 
import copy
import argparse 

plt.ion()


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', help ='Data Folder', type = str)
    parser.add_argument('-device', help = 'Device', type = str)
    args = parser.parse_args()
    return args

def data_transforms(data_dir, device, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]), 
        }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes
    return image_datasets, dataloaders, dataset_sizes, class_names, device

def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std* inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(100)

def main():
    args= build_args()
    image_datasets, dataloaders, dataset_sizes, class_names, device = data_transforms(args.dir, args.device)
    print(dataset_sizes) 
    inputs, classes =  next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)

if __name__ == "__main__":
    main()
    


