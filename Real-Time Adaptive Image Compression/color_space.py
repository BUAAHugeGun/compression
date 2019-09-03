import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def YUV2RGB(self, input):
        x = input.contiguous().view(-1, 3).float()
        mat = torch.tensor([[1.164383, 1.164383, 1.164383],
                            [0, -0.391762, 2.017230],
                            [1.596027, -0.812969, 0]])
        bias = torch.tensor([-16 / 255, -128 / 255, -128 / 255])
        temp = (x + bias).mm(mat)
        return temp.view(input.shape[0], 3, input.shape[2], input.shape[3])

    def RGB2YUV(self, input):
        x = input.contiguous().view(-1, 3).float()
        mat = torch.tensor([[0.256789, -0.148223, 0.439215],
                            [0.504129, -0.290992, -0.367789],
                            [0.097906, 0.439215, -0.071426]])
        bias = torch.tensor([16 / 255, 128 / 255, 128 / 255])
        temp = x.mm(mat) + bias
        return temp.view(input.shape[0], 3, input.shape[2], input.shape[3])

    def YUV2YUV420(self, input):
        return [input[:, 0, :, :].unsqueeze(1), input[:, 1, ::2, ::2].unsqueeze(1), input[:, 2, 1::2, ::2].unsqueeze(1)]

    def YUV4202YUV(self, input):
        up = nn.UpsamplingBilinear2d(scale_factor=2)
        input[1] = up(input[1])
        input[2] = up(input[2])
        return torch.cat(input, 1)

    def forward(self, input, mode):
        if mode == None:
            assert 0
        if mode == 'RGB2YUV':
            return self.RGB2YUV(input)
        if mode == 'YUV2RGB':
            return self.YUV2RGB(input)
        if mode == 'YUV2YUV420':
            return self.YUV2YUV420(input)
        if mode == 'YUV4202YUV':
            return self.YUV4202YUV(input)
        assert 0


if __name__ == '__main__':
    img = cv2.imread('../../test_.png')
    img = np.array(img)
    x = transforms.ToTensor()(img)
    x = x.unsqueeze(0)
    x = x[:, :, 0:16, 0:16]
    t = Model()
    y = x.clone()
    x.requires_grad = True
    x = t.RGB2YUV(x)
    x = t.YUV2YUV420(x)
    x = t.YUV4202YUV(x)
    x = t.YUV2RGB(x)
    print(y - x)
