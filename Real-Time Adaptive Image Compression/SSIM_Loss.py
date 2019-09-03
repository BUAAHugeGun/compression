import torch
import math
import sys
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # gauss.requires_grad = True
    return gauss / gauss.sum()


def create_window(window_size):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
    if torch.cuda.is_available():
        window = window.cuda()
    return window


def ssim(img1, img2, window_size=11, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channels, height, width) = img1.size()
    real_size = min(window_size, height, width)
    window = create_window(real_size)

    ret_channels = []
    cs_channels = []
    for ch in range(channels):  # loop over channels, then average
        img1_ch = torch.unsqueeze(img1[:, ch, :, :], 1)
        img2_ch = torch.unsqueeze(img2[:, ch, :, :], 1)
        mu1 = F.conv2d(img1_ch, window, padding=padd)
        mu2 = F.conv2d(img2_ch, window, padding=padd)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1_ch * img1_ch, window, padding=padd) - mu1_sq
        sigma2_sq = F.conv2d(img2_ch * img2_ch, window, padding=padd) - mu2_sq
        sigma12 = F.conv2d(img1_ch * img2_ch, window, padding=padd) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        cs_channels.append(cs)
        ret_channels.append(ret)

    cs_mean = torch.mean(torch.stack(cs_channels), dim=-1)
    ret_mean = torch.mean(torch.stack(ret_channels), dim=-1)

    if full:
        return ret_mean, cs_mean
    return ret_mean


def msssim(img1, img2, window_size=11, size_average=True, val_range=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # # Normalize (to avoid NaNs)
    #
    # mssim = (mssim + 1) / 2
    # mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # output = torch.prod(pow1 * pow2)
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


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
