import torch
from torch.utils import data
import torch.nn as nn
import torch.functional as F
from GRU import GRU_cell


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bulid()
        # self.initial()

    def bulid(self):
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = GRU_cell(in_channels=64, hidden_channels=256, kernel_size=3, stride=2, padding=1,
                             hidden_size=1,
                             bias=False)
        self.rnn2 = GRU_cell(in_channels=256, hidden_channels=512, kernel_size=3, stride=2, padding=1,
                             hidden_size=1,
                             bias=False)
        self.rnn3 = GRU_cell(in_channels=512, hidden_channels=512, kernel_size=3, stride=2, padding=1,
                             hidden_size=1,
                             bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3
        return x, hidden1, hidden2, hidden3


class Sign(nn.Module):
    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, x):
        if self.training:
            y = x.new(x.size()).uniform_()
            output = x.clone()
            output[(1 - x) / 2 <= y] = 1
            output[(1 - x) / 2 > y] = -1
            return output
        else:
            return x.sign()


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.build()

    def build(self):
        self.sign = Sign()
        self.conv = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv(x)
        x = torch.tanh(y)
        return self.sign(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.build()
        # self.initial()

    def build(self):
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = GRU_cell(in_channels=512, hidden_channels=512, kernel_size=3, stride=1, padding=1, hidden_size=1,
                             bias=False)
        self.rnn2 = GRU_cell(in_channels=128, hidden_channels=512, kernel_size=3, stride=1, padding=1, hidden_size=1,
                             bias=False)
        self.rnn3 = GRU_cell(in_channels=128, hidden_channels=256, kernel_size=3, stride=1, padding=1, hidden_size=3,
                             bias=False)
        self.rnn4 = GRU_cell(in_channels=64, hidden_channels=128, kernel_size=3, stride=1, padding=1, hidden_size=3,
                             bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        y=torch.nn.PixelShuffle(2)
        x = self.conv1(input)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1
        x = y(x)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2
        x = y(x)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3
        x = y(x)

        hidden4 = self.rnn4(x, hidden4)
        x = hidden4
        x = y(x)

        x = torch.tanh(self.conv2(x)) / 2

        return x, hidden1, hidden2, hidden3, hidden4


if __name__ == '__main__':
    print('model test starts')
    decoder = Decoder()
    x = torch.randn(2, 32, 2, 2)
    hidden1 = torch.zeros(2, 512, 2, 2)
    hidden2 = torch.zeros(2, 512, 4, 4)
    hidden3 = torch.zeros(2, 256, 8, 8)
    hidden4 = torch.zeros(2, 128, 16, 16)
    print(decoder(x, hidden1, hidden2, hidden3, hidden4)[0].shape)
