from encoder import Encoder
from decoder import Decoder
from torchvision import transforms
import cv2
import torch
import numpy as np

if __name__=='__main__':
    img_path = '../../test.jpeg'
    img = cv2.imread(img_path)
    x = transforms.ToTensor()(img)
    x = x.unsqueeze(0)
    encoder = Encoder().eval()
    decoder = Decoder().eval()
    x = decoder(encoder(x))
    x=torch.round(x*256).int()
    x=torch.abs(x)
    x = x.detach().numpy()
    x=np.array(x,dtype=np.uint8)
    x=x.squeeze(0)
    x=np.swapaxes(x,0,2)
    x=np.swapaxes(x,0,1)
    cv2.imshow('test', x*5)
    cv2.waitKey(0)

