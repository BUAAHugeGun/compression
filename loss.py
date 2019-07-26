import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, input, target):
        return (input - target).abs().mean()
