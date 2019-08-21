import os
import cv2
import numpy as np
from torchvision import transforms
import pytorch_ssim as SSIM
from PSNR_Loss import Loss as PSNR
import time
if __name__ == '__main__':
    s1 = 0
    s2 = 0
    num = 0
    psnr = PSNR()
    ssim = SSIM.SSIM()
    for maindir, subdir, file_name_list in os.walk("../../dataset/compression/valid"):
        print(len(file_name_list))
        for filename in file_name_list:
            num += 1
            path = os.path.join(maindir, filename)
            print(filename)
            os.system("cwebp -q 90 {} -o output.webp".format(path))
            os.system("dwebp output.webp -o output.png")
            time.sleep(1)
            x = transforms.ToTensor()(np.array(cv2.imread(path)))
            y = transforms.ToTensor()(np.array(cv2.imread("./output.png")))
            x = x.detach()
            x = x.unsqueeze(0)
            y = y.detach()
            y = y.unsqueeze(0)
            s1 += ssim(x, y)
            s2 += psnr(x, y)
            print(s1 / num, s2 / num)
