import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import numpy as np
import os
import glob

class CIFAR100(Dataset):
    def __init__(self, root='../data',train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = datasets.CIFAR100(root=self.root, train=self.train, transform=self.transform, download=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
