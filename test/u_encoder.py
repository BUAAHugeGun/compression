import torch
import torch.nn as nn
import math


class Encoder(nn.Module):
    def __init__(self, out_channels=60):
        super(Encoder, self).__init__()
        self.out_channels = out_channels
        self.build()
        self.initial()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding=0, mode="Avg"):
        if mode != "Max" and mode != "Avg":
            assert 0
        if mode == "Max":
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
        else:
            return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))

    def build(self):
        self.conv11 = self._conv_layer(3, 16, 3, 1, 1)
        self.conv12 = self._conv_layer(16, 16, 3, 1, 1)
        self.pool1 = self._pool_layer(2, 2, 0)  # 64x64

        self.conv21 = self._conv_layer(16, 32, 3, 1, 1)
        self.conv22 = self._conv_layer(32, 32, 3, 1, 1)
        self.pool2 = self._pool_layer(2, 2, 0)  # 32x32

        self.conv31 = self._conv_layer(32, 64, 3, 1, 1)
        self.conv32 = self._conv_layer(64, 64, 3, 1, 1)
        self.pool3 = self._pool_layer(2, 2, 0)  # 16x16

        self.conv41 = self._conv_layer(64, 128, 3, 1, 1)
        self.conv42 = self._conv_layer(128, 128, 3, 1, 1)
        self.pool4 = self._pool_layer(2, 2, 0)  # 8x8

        self.conv51 = self._conv_layer(128, 128, 3, 1, 1)
        self.conv52 = self._conv_layer(128, 128, 3, 1, 1)

        self.up6 = nn.UpsamplingBilinear2d(scale_factor=2)  # 16x16
        self.conv60 = self._conv_layer(128, 128, 3, 1, 1)
        self.conv61 = self._conv_layer(128, self.out_channels, 3, 1, 1)
        self.conv62 = self._conv_layer(self.out_channels, self.out_channels, 3, 1, 1)

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
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.pool3(x)
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.pool4(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.up6(x)
        x = self.conv60(x)
        x = self.conv61(x)
        x = self.conv62(x)
        return x


if __name__ == '__main__':
    x = torch.randn(2, 3, 128, 128)
    test = Encoder(60)
    print(test(x).shape)
