import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset as dataset
from pycocotools.coco import COCO
from torchvision import datasets
from torchvision import transforms
import numpy as np
import cv2


class Dataset(dataset):
    def __init__(self, transform=None, train=True):
        super(Dataset, self).__init__()
        self.transform = transform
        self.set = datasets.ImageFolder('../../dataset/coco/')
        self.data = self.set.imgs
        self.data = np.array(self.data)
        if train:
            self.images = self.data[:, self.set.class_to_idx['train2014']]
        else:
            self.images = self.data[:, self.set.class_to_idx['val2017']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = cv2.imread(self.images[item])
        # img = np.transpose(img, [2, 0, 1])[:, 0:10, 0:10]
        while img.shape[0] < 128:
            img = np.concatenate([img, img], 0)
        while img.shape[1] < 128:
            img = np.concatenate([img, img], 1)
        H, W = img.shape[0] - 128, img.shape[1] - 128
        h, w = random.randint(0, H - 1), random.randint(0, W - 1)
        img = img[h:h + 128, w:w + 128, :]
        return self.transform(img)


if __name__ == '__main__':
    test = Dataset(transform=transforms.Compose([transforms.ToTensor()]))
    print(test.__getitem__(100))
