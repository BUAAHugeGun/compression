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

    def build(self, c):
        self.G = self._conv_layer(96, 96, 3, 1, 1)
        self.g = []
        self.f = []
        self.d = []
        self.g.append(self._deconv_layer(16, c[0], 5, 4, 1))
        self.g.append(self._deconv_layer(16, c[1], 3, 2, 1))
        self.g.append(self._conv_layer(16, c[2], 3, 1, 1))
        self.g.append(self._conv_layer(16, c[3], 3, 1, 1))
        self.g.append(self._conv_layer(16, c[4], 3, 2, 1))
        self.g.append(self._conv_layer(16, c[5], 5, 4, 1))
        for i in range(0, 3):
            f1 = self._deconv_layer(c[i] // 2, 3, 3, 2, 1)
            f2 = self._conv_layer(c[i], c[i] // 2, 3, 1, 1)
            self.f.append(nn.Sequential(f2, f1))
        for i in range(3, self.M):
            f1 = self._conv_layer(c[i] // 2, 3, 3, 1, 1)
            f2 = self._conv_layer(c[i], c[i] // 2, 3, 1, 1)
            self.f.append(nn.Sequential(f2, f1))
        for i in range(0, self.M - 1):
            self.d.append(self._deconv_layer(3, 3, 4, 2, 1))
        self.d_list = nn.Sequential(*self.d)
        self.f_list = nn.Sequential(*self.f)
        self.g_list = nn.Sequential(*self.g)

    def initial(self, scale_factor=1.0, mode="FAN_IN"):
        if mode != "FAN_IN" and mode != "FAN_out":
            assert 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if mode == "FAN_IN":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, y):
        y = self.G(y)
        gy = []
        fy = []
        dy = []
        for i in range(0, self.M):
            gy.append(self.g[i](y[:, i * 16:(i + 1) * 16, :, :]))
        for i in range(0, self.M):
            fy.append(self.f[i](gy[i]))
        for i in range(0, self.M):
            dy.append(fy[i])
        for i in range(1, self.M - 1):
            j = self.M - 1 - i
            dy[j - 1] = (dy[j - 1] + nn.UpsamplingBilinear2d(scale_factor=2)(dy[j])) / 2
        return dy[0]


if __name__ == '__main__':
    x = torch.randn(2, 96, 4, 4)
    test = Decoder()
    print(test(x).shape)
