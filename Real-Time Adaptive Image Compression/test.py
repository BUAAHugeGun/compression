from encoder import Encoder
from decoder import Decoder
from torchvision import transforms
import cv2
import torch
import numpy as np
import os
import Lp_Loss
import pytorch_ssim
from PSNR_Loss import Loss as PSNR


def pad(x):
    b, c, h, w = x.shape
    print(x.shape)
    H = ((h - 1) // 32 + 2) * 32
    W = ((w - 1) // 32 + 2) * 32
    y = torch.zeros([b, c, H, W])
    print(x.shape, y[:, :, 0:h, 0:w].shape)
    y[:, :, 0:h, 0:w] += x
    return y


def load(encoder, decoder):
    encoder.load_state_dict(torch.load('encoder_epoch-9.pth', map_location='cpu'))
    decoder.load_state_dict(torch.load('decoder_epoch-9.pth', map_location='cpu'))


if __name__ == '__main__':
    img_path = '../../test.jpg'
    img = cv2.imread(img_path)
    img = np.array(img)
    print(img)
    x = transforms.ToTensor()(img)
    # x = x[:, 0:128, 0:128]
    x = x.unsqueeze(0)
    # x = x[:, :, 0:128, 0:128]
    b, c, h, w = x.shape
    if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
        x = pad(x)
    encoder = Encoder()
    decoder = Decoder()
    load(encoder, decoder)
    encoder.eval()
    decoder.eval()
    y = x.clone()
    x = decoder(encoder(x))
    l1 = Lp_Loss.Loss(p=1)
    ss = pytorch_ssim.SSIM(window_size=11)
    psnr = PSNR()
    x = torch.clamp(x, 0, 1)
    print(ss(x, y))
    print(l1(x, y))
    print(psnr(x, y))
    x = torch.round(x * 255).int()
    x = torch.abs(x)
    x = x.detach().numpy()
    x = np.array(x, dtype=np.uint8)
    x = x.squeeze(0)
    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)
    x = x[0:w, 0:h, :]
    print(x.shape)
    # cv2.imwrite('./out.jpg', x)
    cv2.waitKey(0)
