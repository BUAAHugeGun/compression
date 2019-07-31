import torch
import torch.nn as nn
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, in_channels=3, M=6):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
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
        if padding == 0:
            output_padding = 0
        else:
            output_padding = 1
        layers.append(nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
            padding=padding, bias=bias, output_padding=output_padding))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def build(self, c):
        self.d = []
        self.f = []
        self.g = []
        for i in range(0, self.M - 1):
            self.d.append(self._conv_layer(3, 3, 4, 2, 1))
        for i in range(0, 3):
            f1 = self._conv_layer(3, c[i] // 2, 3, 2, 1)
            f2 = self._conv_layer(c[i] // 2, c[i], 3, 1, 1)
            self.f.append(nn.Sequential(f1, f2))
        for i in range(3, self.M):
            f1 = self._conv_layer(3, c[i] // 2, 1, 1, 0)
            f2 = self._conv_layer(c[i] // 2, c[i], 1, 1, 0)
            self.f.append(nn.Sequential(f1, f2))
        self.g.append(self._conv_layer(c[0], 16, 5, 4, 1))
        self.g.append(self._conv_layer(c[1], 16, 3, 2, 1))
        self.g.append(self._conv_layer(c[2], 16, 3, 1, 1))
        self.g.append(self._conv_layer(c[3], 16, 3, 1, 1))
        self.g.append(self._deconv_layer(c[4], 16, 3, 2, 1))
        self.g.append(self._deconv_layer(c[5], 16, 5, 4, 1))
        self.G = self._conv_layer(96, 96, 3, 1, 1)
        self.d0 = self.d[0]
        self.d1 = self.d[1]
        self.d2 = self.d[2]
        self.d3 = self.d[3]
        self.d4 = self.d[4]
        self.f0 = self.f[0]
        self.f1 = self.f[1]
        self.f2 = self.f[2]
        self.f3 = self.f[3]
        self.f4 = self.f[4]
        self.f5 = self.f[5]
        self.g0 = self.g[0]
        self.g1 = self.g[1]
        self.g2 = self.g[2]
        self.g3 = self.g[3]
        self.g4 = self.g[4]
        self.g5 = self.g[5]

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

    def forward(self, x):
        dx = []
        dx.append(x)
        fx = []
        gx = []
        for i in range(0, self.M - 1):
            dx.append(self.d[i](dx[i]))
        for i in range(0, self.M):
            fx.append(self.f[i](dx[i]))
            gx.append(self.g[i](fx[i]))
        return self.G(torch.cat(gx, 1))


if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    test = Encoder(3)
    print(test(x).shape)
