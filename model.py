import torch
import torch.nn as nn
import torch.functional as F
from GRU import GRU_cell


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bulid()
        self.initial()

    def GRU(self, in_channels, hidden_channels, kernel_size, stride, padding, hidden_size, bias):
        return GRU_cell(in_channels, hidden_channels, kernel_size, stride, padding, hidden_size, bias)

    def bulid(self):
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = self.GRU(in_channels=64, hidden_channels=256, kernel_size=3, stride=2, padding=1, hidden_size=1,
                        bias=False)
        self.rnn1 = self.GRU(in_channels=256, hidden_channels=512, kernel_size=3, stride=2, padding=1, hidden_size=1,
                        bias=False)
        self.rnn1 = self.GRU(in_channels=512, hidden_channels=512, kernel_size=3, stride=2, padding=1, hidden_size=1,
                        bias=False)
