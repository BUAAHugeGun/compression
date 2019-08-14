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
    encoder.load_state_dict(torch.load('encoder_epoch-25.pth', map_location='cpu'))
    decoder.load_state_dict(torch.load('decoder_epoch-25.pth', map_location='cpu'))


s1 = torch.zeros(1)
s2 = torch.zeros(1)
num = 0


def main(img_path='../../test_.png'):
    global s1, s2, num
    img = cv2.imread(img_path)
    img = np.array(img)
    x = transforms.ToTensor()(img)
    # x = x[:, 0:128 * 10, 0:128 * 10]
    x = x.unsqueeze(0)
    # x = x[:, :, 0:128, 0:128]
    b, c, h, w = x.shape
    if x.shape[2] % 512 != 0 or x.shape[3] % 512 != 0:
        x = pad(x)
    b, c, H, W = x.shape
    x.requires_grad = False
    encoder = Encoder(out_channels=60)
    decoder = Decoder(out_channels=60)
    load(encoder, decoder)
    encoder.eval()
    decoder.eval()
    y = x.clone()
    # print(x.shape)
    z = np.ndarray([b, c, H, W])
    for i in range(0, x.shape[2] // 512):
        for j in range(0, x.shape[3] // 512):
            xx = x[:, :, i * 512:i * 512 + 512, j * 512:j * 512 + 512]
            xx = decoder(encoder(xx).detach())
            z[:, :, i * 512:i * 512 + 512, j * 512:j * 512 + 512] = xx.detach().numpy()
            print(i, j)
    x=torch.tensor(z).float()
    l1 = Lp_Loss.Loss(p=1)
    ss = pytorch_ssim.SSIM(window_size=11)
    psnr = PSNR()
    psnr.eval()
    ss.eval()
    x = torch.clamp(x, 0, 1)
    # print(ss(x, y), l1(x, y), psnr(x, y))
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
    cv2.imwrite('./out.png', x)
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