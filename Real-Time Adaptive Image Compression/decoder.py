import torch
import torch.nn as nn
import numpy as np
import math


class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=60, M=6):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.M = M
        self.in_channels = in_channels
        self.build(c=[128, 64, 64, 64, 64, 64])
        self.initial()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False, k=0):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(k))
        return nn.Sequential(*layers)

    def _deconv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False, k=0):
        layers = []
        layers.append(nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
            padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(k))
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding=0, mode="Avg"):
        if mode != "Max" and mode != "Avg":
            assert 0
        if mode == "Max":
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
        else:
            return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))

    def build(self, c):
        self.G = self._conv_layer(self.out_channels, self.out_channels, 3, 1, 1)
        self.g = []
        self.f = []
        self.d = []
        channels = self.out_channels // 6
        self.g.append(
            nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4), self._conv_layer(channels, c[0], 3, 1, 1, k=0.2)))
        self.g.append(
            nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), self._conv_layer(channels, c[1], 3, 1, 1, k=0.2)))
        self.g.append(self._conv_layer(channels, c[2], 3, 1, 1, k=0.2))
        self.g.append(self._conv_layer(channels, c[3], 3, 1, 1, k=0.2))
        self.g.append(self._conv_layer(channels, c[4], 3, 2, 1, k=0.2))
        self.g.append(self._conv_layer(channels, c[5], 5, 4, 1, k=0.2))
        for i in range(0, 3):
            f1 = self._conv_layer(c[i], self.in_channels, 3, 1, 1)
            f2 = self._conv_layer(c[i], c[i], 3, 1, 1)
            f3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.f.append(nn.Sequential(f3, f2, f1))
        for i in range(3, self.M):
            f1 = self._conv_layer(c[i], self.in_channels, 3, 1, 1)
            f2 = self._conv_layer(c[i], c[i], 3, 1, 1)
            self.f.append(nn.Sequential(f2, f1))
        for i in range(0, self.M - 1):
            self.d.append(nn.Sequential(
                self._conv_layer(self.in_channels * 2, self.in_channels, 3, 1, 1)))
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
            yy = y[:, i * self.out_channels // 6:(i + 1) * self.out_channels // 6, :, :]
            dy.append(self.f[i](self.g[i](yy)))
        for i in range(1, self.M):
            j = self.M - i
            dy[j - 1] = self.d[j - 1](torch.cat([dy[j - 1], nn.UpsamplingNearest2d(scale_factor=2)(dy[j])], 1))
        return dy[0]


if __name__ == '__main__':
    x = torch.randn(1, 30, 160, 320)
    test = Decoder(1, 30)
    print(test(x).shape)
