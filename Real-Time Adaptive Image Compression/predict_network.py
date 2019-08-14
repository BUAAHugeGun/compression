import torch
import torch.nn as nn
import math


class Model(nn.Module):
    def __init__(self, in_channels=60):
        super(Model, self).__init__()
        self.channels = in_channels
        self.build()
        self.initial()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0))
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding=0, mode="Avg"):
        if mode != "Max" and mode != "Avg":
            assert 0
        if mode == "Max":
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
        else:
            return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))

    def build(self):
        self.conv11 = self._conv_layer(self.channels, 64, 3, 1, 1)
        self.conv12 = self._conv_layer(64, 64, 3, 1, 1)  # 16x16
        self.pool1 = self._pool_layer(2, 2)  # 8x8

        self.conv21 = self._conv_layer(64, 128, 3, 1, 1)
        self.conv22 = self._conv_layer(128, 128, 3, 1, 1)  # 8x8
        self.pool2 = self._pool_layer(2, 2)  # 4x4

        self.conv31 = self._conv_layer(128, 256, 3, 1, 1)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)  # 8x8
        self.conv41 = self._conv_layer(256, 128, 3, 1, 1)
        self.conv42 = self._conv_layer(256, 128, 3, 1, 1)

        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)  # 16x16
        self.conv51 = self._conv_layer(128, 64, 3, 1, 1)
        self.conv52 = self._conv_layer(128, 64, 3, 1, 1)
        self.conv53 = self._conv_layer(64, self.channels, 3, 1, 1)

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
        x1 = x.clone()
        x = self.pool1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x2 = x.clone()
        x = self.pool2(x)

        x = self.conv31(x)

        x = self.up4(x)
        x = self.conv41(x)
        x = torch.cat([x2, x], 1)
        x = self.conv42(x)

        x = self.up5(x)
        x = self.conv51(x)
        x = torch.cat([x1, x], 1)
        x = self.conv52(x)
        return self.conv53(x)

if __name__=='__main__':
    x=torch.randn(2,60,16,16)
    test=Model()
    torch.save(test.state_dict(),'./test.pth')
    print(test(x).shape)
