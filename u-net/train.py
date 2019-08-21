import torch
import torch.nn as nn
import torchvision
import argparse
import os
from torch.utils.data import DataLoader
from u_decoder import Decoder
from u_encoder import Encoder
from predict_network import Model as PD
from quantization import Quantizator
# from SSIM_Loss import Loss
from dataset import Dataset
from torchvision import transforms
import pytorch_ssim
from Lp_Loss import Loss as lp


# from Lp_Loss import Loss
def load(encoder, decoder, predictor, log_dir, epoch):
    pre_encoder = torch.load('./logs/encoder_epoch-{}.pth'.format(epoch), map_location='cpu')
    now_encoder = encoder.state_dict()
    pre_encoder = {k: v for k, v in pre_encoder.items() if k in now_encoder}
    now_encoder.update(pre_encoder)
    encoder.load_state_dict(now_encoder)

    pre_decoder = torch.load('./logs/decoder_epoch-{}.pth'.format(epoch), map_location='cpu')
    now_decoder = decoder.state_dict()
    pre_decoder = {k: v for k, v in pre_decoder.items() if k in now_decoder}
    now_decoder.update(pre_decoder)
    decoder.load_state_dict(now_decoder)

    if os.path.exists('./logs/predictor_epoch-{}.pth'):
        predictor.load_state_dict(torch.load('./logs/predictor_epoch-{}.pth'.format(epoch), map_location='cpu'))


def train(encoder, decoder, predictor, train_loader, test_loader, opt, pre_opt, sch, pre_sch, criterion, args):
    encoder.train()
    decoder.train()
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        predictor = predictor.cuda()
    if args.load_epoch >= 0:
        load(encoder, decoder, predictor, args.log_dir, args.load_epoch)
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log = open(os.path.join(log_dir, "log.txt"), 'w')
    print(args, file=log)

    tot = 0
    l1 = lp()
    for epoch in range(args.epoch):
        sch.step()
        # pre_sch.step()
        for id, data in enumerate(train_loader):
            tot += 1
            img = data.clone()
            if torch.cuda.is_available():
                img = img.cuda()
                data = data.cuda()
            quantizator = Quantizator()
            x = encoder(img)
            """
            y = x.clone().detach()
            x = quantizator(x)
            z = x.clone().detach()
            p = predictor(z)
            loss_pre1 = l1(p, y)
            loss_pre2 = 1 - criterion(p, y)
            loss_pre = loss_pre1
            print(l1(y, z), p.sum(), torch.abs(p).sum())
            x += p.detach()
            """

            output = decoder(x)
            loss1 = 1 - criterion(output, data)
            loss2 = l1(output, data)
            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_pre1 = torch.tensor([0])
            loss_pre2 = torch.tensor([0])
            # pre_opt.zero_grad()
            # loss_pre.backward()
            # pre_opt.step()

            if tot % args.show_interval == 0:
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss:{:.7f},{:.7f},{:.7f},{:.7f}]\t'.format(loss1.item(), loss2.item(), loss_pre1.item(),
                                                                  loss_pre2.item()),
                    '[lr:{:.7f},{:7f}]'.format(sch.get_lr()[0], pre_sch.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss:{:.7f},{:.7f},{:.7f},{:.7f}]\t'.format(loss1.item(), loss2.item(), loss_pre1.item(),
                                                                  loss_pre2.item()),
                    '[lr:{:.7f},{:7f}]'.format(sch.get_lr()[0], pre_sch.get_lr()[0]),
                    file=log
                )
        if epoch % args.snapshot_interval == 0:
            torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder_epoch-{}.pth'.format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(log_dir, 'decoder_epoch-{}.pth'.format(epoch)))
            torch.save(predictor.state_dict(), os.path.join(log_dir, 'predictor_epoch-{}.pth'.format(epoch)))

    log.close()


def main(args):
    encoder = Encoder(out_channels=30)
    decoder = Decoder(out_channels=30)
    predictor = PD(30)
    criterion = pytorch_ssim.SSIM(window_size=11)

    opt = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=args.lr)
    pre_opt = torch.optim.Adam([{'params': predictor.parameters()}], lr=args.lr * 2)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, args.lr_milestion, gamma=0.5)
    pre_sch = torch.optim.lr_scheduler.MultiStepLR(pre_opt, args.lr_milestion, gamma=0.5)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = Dataset(train=True, transform=transform)
    test_set = Dataset(train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    train(encoder, decoder, predictor, train_loader, test_loader, opt, pre_opt, sch, pre_sch, criterion, args)


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--root', default='./')
    paser.add_argument('--log_dir', default='./logs')
    paser.add_argument('--batch_size', default=16)
    paser.add_argument('--num_workers', default=2)
    paser.add_argument('--lr', default=0.001)
    paser.add_argument('--lr_milestion', default=[5, 20, 100, 500])
    paser.add_argument('--epoch', default=2000)
    paser.add_argument('--show_interval', default=1)
    paser.add_argument('--test_interval', default=2)
    paser.add_argument('--snapshot_interval', default=5)
    paser.add_argument('--load_epoch', default=990)
    args = paser.parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    print(args)
    main(args)
