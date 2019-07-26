import torch
import os
import time
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import model
from loss import Loss
from dataset import Dataset


def train(args):
    transform = transforms.Compose([transforms.RandomCrop(32, 32), transforms.ToTensor])
    trainset = Dataset(transform)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model_dir = args.log_model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log = open(os.path.join(model_dir, "log.txt"), "w")
    print(args, file=log)

    encoder = model.Encoder()
    binarizer = model.Binarizer()
    decoder = model.Decoder()
    opt = torch.optim.Adam(
        [{'params': encoder.parameters()}, {'params': binarizer.parameters()}, {'params': decoder.parameters()}],
        lr=args.lr)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    tot = 0
    loss = Loss()
    for epoch in range(args.epoch):
        sch.step()
        for id, data in enumerate(train_loader):
            tot += 1
            t0 = time.time()
            data=torch.randn(data.size(0),3,32,32)
            encoder_h_1 = torch.zeros(data.size(0), 256, 8, 8)
            encoder_h_2 = torch.zeros(data.size(0), 512, 4, 4)
            encoder_h_3 = torch.zeros(data.size(0), 512, 2, 2)

            decoder_h_1 = torch.zeros(data.size(0), 512, 2, 2)
            decoder_h_2 = torch.zeros(data.size(0), 512, 4, 4)
            decoder_h_3 = torch.zeros(data.size(0), 256, 8, 8)
            decoder_h_4 = torch.zeros(data.size(0), 128, 16, 16)
            opt.zero_grad()
            loss_sum = 0
            res = data - 0.5
            for _ in range(args.iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                codes = binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                res = res - output
                print(res.abs.mean())
                loss_sum += res.abs().mean()
            loss_sum /= args.iterations
            loss_sum.backward()
            opt.step()
            t1 = time.time()
            print('one image time:{}'.format(t1 - t0))

            if tot % args.show_interval == 0:
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss: {:.3f}]\t'.format(loss_sum),
                    '[lr: {:.6f}]'.format(sch.get_lr()[0])
                )
                print(
                    '[epoch:{}, batch:{}]\t'.format(epoch, id),
                    '[loss: {:.3f}]\t'.format(loss_sum),
                    '[lr: {:.6f}]'.format(sch.get_lr()[0]),
                    file=log
                )
        """
        if epoch % args.test_interval == 0:
            loss_sum = 0.
            acc_sum = 0.
            test_batch_num = 0
            total_num = 0
            for idx, data in enumerate(test_loader):
                test_batch_num += 1
                img, label = data
                total_num += img.shape[0]
                if torch.cuda.is_available():
                    img, label = img.cuda(), label.cuda()
                output = net(img)

                loss = criterion(output, label)
                loss_sum += loss.item()
                acc_sum += torch.eq(torch.max(output, dim=1)[1], label).sum().cpu().float()
            print('\n***************validation result*******************')
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}'.format(acc_sum / total_num)
            )
            print('****************************************************\n')
            print('\n***************validation result*******************', file=log)
            print(
                'loss_avg: {:.3f}\t'.format(loss_sum / test_batch_num),
                'accuracy_avg: {:.3f}'.format(acc_sum / total_num),
                file=log
            )
            print('****************************************************\n', file=log)
            """

        if epoch % args.snapshot_interval == 0:
            model_path = './model'
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder_epoch{}.pth'.format(epoch)))
            torch.save(binarizer.state_dict(), os.path.join(model_path, 'binarizer_epoch{}.pth'.format(epoch)))
            torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder_epoch{}.pth'.format(epoch)))
    log.close()


def main(args):
    train(args)


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument("--root", default="./")
    paser.add_argument("--log_model_dir", default="./log")
    paser.add_argument("--batch_size", default=25)
    paser.add_argument('--num_workers', default=2)
    paser.add_argument("--lr", default=0.0001)
    paser.add_argument("--epoch", default=10)
    paser.add_argument("--evaluate", default=False)
    paser.add_argument("--show_interval", default=1)
    paser.add_argument("--test_interval", default=2)
    paser.add_argument("--snapshot_interval", default=5)
    paser.add_argument("--iterations", default=16)
    args = paser.parse_args()
    if not os.path.exists(args.log_model_dir):
        os.mkdir(args.log_model_dir)
    print(args)
    main(args)
