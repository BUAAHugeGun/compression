import torch
import torch.nn as nn
import numpy as np
import math


class Decoder(nn.Module):
    def __init__(self, out_channels=3, M=6):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.M = M
        self.build(c=[256, 128, 64, 32, 16, 8])
        self.initial()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _deconv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=True):
        layers = []
        layers.append(nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
            padding=padding, bias=bias, output_padding=1))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def build(self):
        self.G=self._deconv_layer(96,96,3,1,1)