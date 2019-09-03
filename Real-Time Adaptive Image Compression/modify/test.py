from encoder import Encoder
from decoder import Decoder
from torchvision import transforms
from predict_network import Model as Predictor
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


def load(encoder, decoder, pre=None):
    pre_encoder = torch.load('./logs/encoder_epoch-{}.pth'.format(2000), map_location='cpu')
    now_encoder = encoder.state_dict()
    pre_encoder = {k[7:]: v for k, v in pre_encoder.items()}
    now_encoder.update(pre_encoder)
    encoder.load_state_dict(now_encoder)

    pre_decoder = torch.load('./logs/decoder_epoch-{}.pth'.format(2000), map_location='cpu')
    now_decoder = decoder.state_dict()
    pre_decoder = {k[7:]: v for k, v in pre_decoder.items()}
    now_decoder.update(pre_decoder)
    decoder.load_state_dict(now_decoder)


s1 = torch.zeros(1)
s2 = torch.zeros(1)
num = 0

pic = []

encoder = None
decoder = None
predictor = None


def run(x):
    global pic, encoder, decoder, predictor
    print(x.shape)
    x.requires_grad = False
    with torch.no_grad():
        y = encoder(x)
        #p = y.squeeze(0).clone()

        #p = torch.round(p * 255).int()
        #p = np.array(p, dtype=np.uint8)

        x = decoder(y)
    return x


def main(img_path='../../test_.png', base=256):
    global s1, s2, num, pic, encoder, decoder, predictor
    num += 1
    encoder = Encoder(out_channels=30)
    decoder = Decoder(out_channels=30)
    predictor = Predictor(30)
    load(encoder, decoder, predictor)
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
    x = run(x)
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
    x = x.detach().cpu().numpy()
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
    test()
