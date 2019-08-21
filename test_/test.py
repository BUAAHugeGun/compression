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
    encoder.load_state_dict(torch.load('encoder_epoch-1995.pth', map_location='cpu'))
    decoder.load_state_dict(torch.load('decoder_epoch-1995.pth', map_location='cpu'))


s1 = torch.zeros(1)
s2 = torch.zeros(1)
num = 0

pic = []

encoder = None
decoder = None


def run(x):
    global pic, encoder, decoder
    print(x.shape)
    te = nn.ReflectionPad2d(32)
    x = te(x)
    x.requires_grad = False
    with torch.no_grad():
        y = encoder(x)
        pic.append(y.clone())
        x = decoder(y).detach()
        x = x[:, :, 32:-32, 32:-32]
    return x


def main(img_path='../../test_.png', base=256):
    global s1, s2, num, pic, encoder, decoder
    num += 1
    encoder = Encoder(out_channels=60)
    decoder = Decoder(out_channels=60)
    load(encoder, decoder)
    encoder.eval()
    decoder.eval()
    img = cv2.imread(img_path)
    img = np.array(img)
    x = transforms.ToTensor()(img)
    # x = x[:, 0:128 * 10, 0:128 * 10]
    x = x.unsqueeze(0)
    # x = x[:, :, 0:128, 0:128]
    b, c, h, w = x.shape
    if x.shape[2] % base != 0 or x.shape[3] % base != 0:
        x = pad(x, base)
    b, c, H, W = x.shape
    x.requires_grad = False
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        x = x.cuda()
    y = x.clone()

    if torch.cuda.is_available():
        pass
    z = []
    for i in range(0, x.shape[2] // base):
        for j in range(0, x.shape[3] // base):
            z.append(x[:, :, i * base:(i + 1) * base, j * base:(j + 1) * base])

    z = torch.cat(z, 0)
    patches = z.shape[0]
    print(patches)

    z[0:patches // 4, :, :, :] = run(z[0:patches // 4, :, :, :]).detach()
    z[patches // 4:patches // 2, :, :, :] = run(z[patches // 4:patches // 2, :, :, :]).detach()
    z[patches // 2:patches * 3 // 4, :, :, :] = run(z[patches // 2:patches * 3 // 4, :, :, :]).detach()
    z[patches * 3 // 4:patches, :, :, :] = run(z[patches * 3 // 4:patches, :, :, :]).detach()

    """
    pic = torch.cat(pic, 0) * 255
    print(pic.shape)
    pic = pic.reshape(-1, 36*30, 3).int().numpy()
    cv2.imwrite('test.png', pic)
    """

    tot = 0
    for i in range(0, x.shape[2] // base):
        for j in range(0, x.shape[3] // base):
            x[:, :, i * base:(i + 1) * base, j * base:(j + 1) * base] = z[tot]
            tot += 1
    l1 = Lp_Loss.Loss(p=1)
    ss = pytorch_ssim.SSIM(window_size=11)
    psnr = PSNR()
    psnr.eval()
    ss.eval()
    x = torch.clamp(x, 0, 1)
    s1 += ss(x, y)
    s2 += psnr(x, y)
    x = x.detach()
    y = y.detach()
    s1 = s1.detach()
    s2 = s2.detach()
    print(s1 / num, s2 / num)
    x = torch.round(x * 255).int()
    x = torch.abs(x)
    x = x.detach().numpy()
    x = np.array(x, dtype=np.uint8)
    x = x.squeeze(0)
    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)
    x = x[0:h, 0:w, :]
    cv2.imwrite('./out__.png', x)


def test():
    for maindir, subdir, file_name_list in os.walk("../../dataset/compression/valid"):
        for filename in file_name_list:
            print(filename)
            main(os.path.join(maindir, filename))


if __name__ == '__main__':
    import os

    import time

    x=torch.randn(1,3,512,512)
    encoder=Encoder(out_channels=30)
    decoder=Decoder(30)
    if torch.cuda.is_available():
        encoder=encoder.cuda()
        decoder=decoder.cuda()
        x=x.cuda()
    x=decoder(encoder(x.detach()))
