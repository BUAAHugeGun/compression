import torch
import torch.nn as nn
import os
import numpy
from torch.utils.data import Dataset as torchdata
from torchvision import datasets
from torchvision import transforms
from pycocotools.coco import COCO
#coco=COCO('../dataset/coco/val2017.zip')

class Dataset(torchdata):
    def __init__(self, transform=None, train=True):
        super(Dataset, self).__init__()
        self.transform = transform
        self.data=datasets.ImageFolder('../dataset/coco/')
        """
        if train == True:
            self.set = datasets.MNIST(root="../practice/mnist", train=train, transform=transforms.ToTensor(),
                                      download=True)
        else:
            self.set = datasets.MNIST(root="../practice/mnist", train=False, transform=transforms.ToTensor(),
                                      download=True)
        """

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, x):
        img=self.data.__getitem__(x)[0]
        return self.transform(img)


if __name__ == '__main__':
    print('test of dataset starts')
    test = Dataset(train=True,transform=transforms.Compose([transforms.ToTensor()]))
    print(test.__getitem__(0).shape)
