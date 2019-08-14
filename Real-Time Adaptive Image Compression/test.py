from encoder import Encoder
from decoder import Decoder
from torchvision import transforms
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import Lp_Loss
import pytorch_ssim
from PSNR_Loss import Loss as PSNR


def pad(x, base=32):
    b, c, h, w = x.shape
    H = (h // base + 1) * base
    W = (w // base + 1) * base
    y = torch.zeros([b, c, H, W])
    if x.dtype == torch.int:
        y = y.int()
    y[:, :, 0:h, 0:w] += x
    return y


def load(encoder, decoder):
    encoder.load_state_dict(torch.load('encoder_epoch-990.pth', map_location='cpu'))
    decoder.load_state_dict(torch.load('decoder_epoch-990.pth', map_location='cpu'))


s1 = torch.zeros(1)
s2 = torch.zeros(1)
num = 0


def main(img_path='../../test.jpeg'):
    global s1, s2, num
    img = cv2.imread(img_path)
    img = np.array(img)
    x = transforms.ToTensor()(img)
    # x = x[:, 0:128 * 10, 0:128 * 10]
    x = x.unsqueeze(0)
    # x = x[:, :, 0:128, 0:128]
    b, c, h, w = x.shape
    if x.shape[2] % 256 != 0 or x.shape[3] % 256 != 0:
        x = pad(x, 256)
    b, c, H, W = x.shape
    x.requires_grad = False
    encoder = Encoder(out_channels=30)
    decoder = Decoder(out_channels=30)
    load(encoder, decoder)
    encoder.eval()
    decoder.eval()
    y = x.clone()
    # print(x.shape)
    te = nn.ReflectionPad2d(48)
    x = te(x)
    z = np.ndarray(x.shape)
    for i in range(0, x.shape[2] // 256):
        for j in range(0, x.shape[3] // 256):
            xx = x[:, :, i * 256:i * 256 + 256 + 96, j * 256:j * 256 + 256 + 96]
            xx = decoder(encoder(xx).detach())[:, :, 48:48 + 256, 48:48 + 256]
            z[:, :, i * 256 + 48:i * 256 + 256 + 48, j * 256 + 48:j * 256 + 256 + 48] = xx.detach().numpy()
            print(i, j)
    x = torch.tensor(z).float()
    x = x[:, :, 48:H + 48, 48:W + 48]
    l1 = Lp_Loss.Loss(p=1)
    ss = pytorch_ssim.SSIM(window_size=11)
    psnr = PSNR()
    psnr.eval()
    ss.eval()
    x = torch.clamp(x, 0, 1)
    # print(ss(x, y), l1(x, y), psnr(x, y))
    print(x.shape, y.shape)
    s1 += ss(x, y)
    s2 += psnr(x, y)
    x = x.detach()
    y = y.detach()
    s1 = s1.detach()
    s2 = s2.detach()
    print(s1, s2)
    x = torch.round(x * 255).int()
    x = torch.abs(x)
    x = x.detach().numpy()
    x = np.array(x, dtype=np.uint8)
    x = x.squeeze(0)
    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)
    x = x[0:h, 0:w, :]
    cv2.imwrite('./out__.png', x)
    cv2.waitKey(0)


if __name__ == '__main__':
    import os

    main()
    '''
    for maindir, subdir, file_name_list in os.walk("../../dataset/compression/valid"):
        for filename in file_name_list:
            print(filename)
            num += 1
            main(os.path.join(maindir, filename))
    '''
