import torch
import torch.nn as nn
import os
import numpy
from torch.utils.data import Dataset as torchdata
from torchvision import datasets
from torchvision import transforms

class Dataset(torchdata):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        if train == True:
            self.set = datasets.MNIST(root="../practice/mnist", train=train, transform=transforms.ToTensor(), download=True)
        else:
            self.set = datasets.MNIST(root="../practice/mnist", train=False, transform=transforms.ToTensor(), download=True)

    def __len__(self):
        return self.set.__len__()

    def __getitem__(self, x):
        return torch.stack([self.set.__getitem__(x)[0][0],self.set.__getitem__(x)[0][0],self.set.__getitem__(x)[0][0]],0)


if __name__=='__main__':
    print('test of dataset starts')
    test=Dataset(train=True)
    print(test.__getitem__(0).shape)