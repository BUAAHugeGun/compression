import torch
import torch.nn as nn


class Quantizator(nn.Module):
    def __init__(self, B=6):
        super(Quantizator, self).__init__()
        self.B = B - 1

    def forward(self, y):
        factor = 1 << self.B
        return torch.ceil(y * factor) / factor


if __name__ == '__main__':
    x = torch.randn(2, 1, 2, 2)
    print(x)
    test = Quantizator()
    print(test(x))
