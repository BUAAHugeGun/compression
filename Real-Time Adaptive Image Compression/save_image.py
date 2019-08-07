import torch
import torch.nn as nn
import adaptive_arithmetic_coder as ACC
from bit_plane import Bit_plane as BP
from test import pad
import struct


def save(image, B=6, base=8):
    file = open('./image.wm', 'wb')
    if image.shape[0] != 1:
        assert 0
    image = image.squeeze(0)
    if image.shape[2] % base != 0 or image.shape[3] % base != 0:
        image = pad(image, base=base)
    for i in range(0, image.shape[0]):
        for b in range(0, B):
            for j in range(0, image.shape[2] // base):
                for k in range(0, image.shape[3] // base):
                    x = image[i, b, j:j + base, k:k + base].contiguous().view(-1)
                    x = x.tolist()
                    x = "".join([str(m) for m in x])
                    # print(x)
                    x = ACC.encode(x)[0]
                    # print(x)
                    y = bytearray(8)
                    for p in range(len(x) // base):
                        y[p] = int(x[p * base:base * (p + 1)], 2)
                    print(y)
                    file.write(y)


if __name__ == '__main__':
    from quantization import Quantizator

    quan = Quantizator()
    test = BP()
    x = torch.abs(torch.randn(1, 1, 8, 8))
    x = quan(x)
    x = test(x)
    save(x)
    print("")
    file = open('image.wm', "rb")
    y = bytearray
    y = file.read()
    print(y)
