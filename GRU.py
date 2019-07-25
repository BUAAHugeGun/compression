import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class GRU_cell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, hidden_size,
                 dilation=1, bias=True):
        super(GRU_cell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_size = hidden_size
        self.dilation = dilation
        self.bias = bias
        self.gate_channels = 3 * self.hidden_size
        self.build()
        self.initial()

    def build(self):
        self.convih = nn.Conv2d(in_channels=self.in_channels, out_channels=self.gate_channels,
                                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, bias=self.bias)
        self.convhh = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.gate_channels,
                                kernel_size=self.hidden_size, stride=1, padding=self.hidden_size // 2,
                                dilation=1, bias=self.bias)

    def initial(self):
        self.convhh.reset_parameters()
        self.convih.reset_parameters()

    def forward(self, x, hidden):
        gate_x = self.convih(x)
        gate_h = self.convhh(hidden)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (reset_gate * h_n))
        y = newgate + input_gate * (hidden - newgate)
        return y


if __name__ == '__main__':
    test = GRU_cell(in_channels=3, hidden_channels=64, kernel_size=3, stride=2, padding=1, hidden_size=1)
    x=torch.randn([1,3,9,9])
    hidden=torch.randn([1,64,5,5])
    print(test(x,hidden).shape)
