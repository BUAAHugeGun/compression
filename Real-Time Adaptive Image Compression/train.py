import torch
import torch.nn as nn
import torchvision
import argparse
import os
from torch.utils.data import DataLoader
from decoder import Decoder
from encoder import Encoder
from qauntization import Quantizator
# from SSIM_Loss import Loss
from dataset import Dataset
from torchvision import transforms
import pytorch_ssim
from Lp_Loss import Loss as lp


# from Lp_Loss import Loss
def load(encoder, decoder, log_dir, epoch):
    encoder.load_state_dict(torch.load('encoder_epoch-{}.pth'.format(epoch), map_location='cpu'))
    decoder.load_state_dict(torch.load('decoder_epoch-{}.pth'.format(epoch), map_location='cpu'))


def train(encoder, decoder, train_loader, test_loader, opt, sch, criterion, args):
    encoder.train()
    decoder.train()
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    if args.load_epoch >= 0:
        load(encoder, decoder, args.log_dir, args.load_epoch)
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log = open(os.path.join(log_dir, "log.txt"), 'w')
    print(args, file=log)

    tot = 0
    for epoch in range(args.epoch):
        sch.step()
        for id, data in enumerate(train_loader):
            tot += 1
            img = data.clone()
            if torch.cuda.is_available():
                img = img.cuda()
                data = data.cuda()
            quantizator = Quantizator()
            y = encoder(img)
            y = quantizator(y)
            output = decoder(y)
            l1 = lp()
            loss1 = 1 - criterion(output, data)
            loss2 = l1(output, data)
            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()

            if tot % args.show_interval == 0:
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss:{:.7f},{:.7f}]\t'.format(loss1.item(), loss2.item()),
                    '[lr:{:.7f}]'.format(sch.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss:{:.7f},{:.7f}]\t'.format(loss2.item(), loss2.item()),
                    '[lr:{:.7f}]'.format(sch.get_lr()[0]),
                    file=log
                )
        if epoch % args.snapshot_interval == 0:
            torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder_epoch-{}.pth'.format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(log_dir, 'decoder_epoch-{}.pth'.format(epoch)))

    log.close()


def main(args):
    encoder = Encoder(in_channels=3, M=6, out_channels=60)
    decoder = Decoder(out_channels=60, M=6)
    criterion = pytorch_ssim.SSIM(window_size=11)

    opt = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=args.lr)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, args.lr_milestion, gamma=0.5)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = Dataset(train=True, transform=transform)
    test_set = Dataset(train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    train(encoder, decoder, train_loader, test_loader, opt, sch, criterion, args)


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--root', default='./')
    paser.add_argument('--log_dir', default='./logs')
    paser.add_argument('--batch_size', default=16)
    paser.add_argument('--num_workers', default=2)
    paser.add_argument('--lr', default=0.0003)
    paser.add_argument('--lr_milestion', default=[3, 7])
    paser.add_argument('--epoch', default=10)
    paser.add_argument('--show_interval', default=1)
    paser.add_argument('--test_interval', default=2)
    paser.add_argument('--snapshot_interval', default=1)
    paser.add_argument('--load_epoch', default=-1)
    args = paser.parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    print(args)
    main(args)
