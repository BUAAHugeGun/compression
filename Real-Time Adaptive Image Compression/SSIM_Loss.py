import torch
import math
import sys
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Loss(nn.Module):
    def __init__(self, SSIM_c1=0.002, SSIM_c2=0.002, SSIM_c3=0.001, arfa=1, beta=1, gamma=1, k_L1=1, k_SSIM=1):
        super(Loss, self).__init__()
        self.SSIM_c1 = SSIM_c1
        self.SSIM_c2 = SSIM_c2
        self.SSIM_c3 = SSIM_c3
        self.arfa = arfa
        self.beta = beta
        self.gamma = gamma
        self.k_L1 = k_L1
        self.k_SSIM = k_SSIM

    def SSIM(self, input, target):
        if input.shape != target.shape:
            sys.stderr.write("expected input size :{} but get size :{}".format(target.shape, input.shape))
            assert 0
        n_input = input.shape[1] * input.shape[2] * input.shape[3]
        n_target = target.shape[1] * target.shape[2] * target.shape[3]
        mu_x = input.mean()
        mu_y = target.mean()
        sigma_x = input.std()
        sigma_y = target.std()
        sigma2_xy = (((input - mu_x) * (target - mu_y))).mean()
        sigma2_xy = ((input * target) - mu_x * mu_y).sum() / (n_input - 1)
        print(sigma2_xy, sigma_x ** 2 + sigma_y ** 2)
        SSIM_L = (2. * mu_x * mu_x + self.SSIM_c1) / (mu_x * mu_x + mu_y * mu_y + self.SSIM_c1)
        SSIM_C = (2. * sigma2_xy + self.SSIM_c2) / (sigma_x * sigma_x + sigma_y * sigma_y + self.SSIM_c2)
        SSIM_S = (sigma2_xy + self.SSIM_c3) / (sigma_x * sigma_y + self.SSIM_c3)
        # return (SSIM_L ** self.arfa) * (SSIM_C ** self.beta) * (SSIM_S ** self.gamma)
        return (2. * mu_x * mu_y + self.SSIM_c1) * (2. * sigma2_xy + self.SSIM_c2) / (
                mu_x ** 2 + mu_y ** 2 + self.SSIM_c1) / (sigma_y ** 2 + sigma_x ** 2 + self.SSIM_c2)

    def forward(self, input, target):
        return 1 - self.SSIM(input, target)


if __name__ == "__main__":
    x = [[1, -2], [-2, 1]]
    y = [[-1, 2], [2, -1]]
    x = np.array(x)
    y = np.array(y)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    test = Loss()
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    y = y.unsqueeze(0)
    print(test(x, y))
    print(test(x, x))
