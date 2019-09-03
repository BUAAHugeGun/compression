import torch
import torch.nn as nn
import torchvision
import argparse
import os
from torch.utils.data import DataLoader
from decoder import Decoder
from encoder import Encoder
from predict_network import Model as PD
from quantization import Quantizator
# from SSIM_Loss import Loss
from dataset import Dataset
from torchvision import transforms
import pytorch_ssim
from SSIM_Loss import msssim
from TV_Loss import Loss as TV
from Lp_Loss import Loss as lp
import color_space


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
    tv = TV()
    color = color_space.Model()
    for epoch in range(args.epoch):
        sch.step()
        # pre_sch.step()
        for id, data in enumerate(train_loader):
            tot += 1
            img = data.clone()
            img.requires_grad = True
            data.requires_grad = True
            if torch.cuda.is_available():
                img = img.cuda()
                data = data.cuda()

            """
            data_ = color(data, 'RGB2YUV')
            data_ = color(data_, 'YUV2YUV420')
            output = []
            for k in range(0, 3):
                output.append(decoder(encoder(data_[k])))
            output = color(output, 'YUV4202YUV')
            output = color(output, 'YUV2RGB')
            """
            output = decoder(encoder(data))

            output = output.clamp(0, 1)
            loss1 = 1 - msssim(output, data)
            loss2 = l1(output, data)
            loss3 = tv(output)
            loss = loss1
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
                    '[loss:{:.7f},{:.7f},{:.7f},{:.7f}]\t'.format(loss1.item(), loss2.item(), loss3.item(),
                                                                  loss_pre2.item()),
                    '[lr:{:.7f},{:7f}]'.format(sch.get_lr()[0], pre_sch.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss:{:.7f},{:.7f},{:.7f},{:.7f}]\t'.format(loss1.item(), loss2.item(), loss3.item(),
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
    encoder = Encoder(in_channels=3, out_channels=60)
    decoder = Decoder(in_channels=3, out_channels=60)
    torch.save(encoder.state_dict(), './test.pth')
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
    paser.add_argument('--lr', default=0.0003)
    paser.add_argument('--lr_milestion', default=[10, 30, 60, 100, 1000])
    paser.add_argument('--epoch', default=2001)
    paser.add_argument('--show_interval', default=1)
    paser.add_argument('--test_interval', default=2)
    paser.add_argument('--snapshot_interval', default=5)
    paser.add_argument('--load_epoch', default=-1)
    args = paser.parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    print(args)
    main(args)
