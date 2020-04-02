from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
from glob import glob

from data.CelebA import get_dataset
from torch.utils.data import DataLoader
from loss.ssim_loss import SSIMLoss
from utils.weights_init import weights_init
from loss.kl_loss import KLLoss
from loss.pytorch_spl_loss import SPLoss
from model.vae import VAE

parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--nc", type=int, default=3, help="input image channels")
parser.add_argument("--fmaps", type=int, default=64, help="features maps channels")
parser.add_argument("--latent_variable_size", type=int, default=500)
parser.add_argument("--rec_loss", type=str, default="ssim", choices=("ssim", "bce", "l1", "spl"))
parser.add_argument("--size_average", action="store_true", default=False, help="")
parser.add_argument("--resume", action="store_true", help="continue to train")
parser.add_argument("--window_size", type=int, default=11, help="the window size of ssim")
parser.add_argument("--ssim_method", type=str, default="lcs")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = DataLoader(get_dataset(dsize=args.size), batch_size=args.batch_size,  num_workers= 8,shuffle=True,pin_memory=True)
test_loader = DataLoader(get_dataset(root="/mnt/mfs/yiling/new_EL_surface/test", dsize=args.size), batch_size=args.batch_size, shuffle=True)

model = VAE(args)
model.apply(weights_init())
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

if args.rec_loss == "ssim":
    rec_loss = SSIMLoss(method=args.ssim_method)
elif args.rec_loss == "spl":
    rec_loss = SPLoss()
elif args.rec_loss == "bce":
    rec_loss = nn.BCELoss(size_average=args.size_average)
elif args.rec_loss == "l1":
    rec_loss = nn.L1Loss(size_average=args.size_average)
else:
    rec_loss = None
kl_loss = KLLoss(size_average=args.size_average)


def loss_function(x, x_rec, mu, logvar):
    return rec_loss(x, x_rec), kl_loss(mu, logvar)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if data.shape[0] != args.batch_size:
            continue
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recLoss,klLoss = loss_function(recon_batch, data, mu, logvar)
        loss = recLoss+klLoss
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            test(batch_idx)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\trecLoss: {:.6f}\tklLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*128),
                100. * batch_idx / len(train_loader),
                recLoss.data.item() / len(data),klLoss.data.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*128)))
    return train_loss / (len(train_loader)*128)


def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        if data.shape[0] != args.batch_size:
            continue
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        recLoss,klLoss = loss_function(recon_batch, data, mu, logvar)
        test_loss += (recLoss+klLoss).data.item()

        torchvision.utils.save_image(data.data, '../imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch.data, '../imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

    test_loss /= (len(test_loader)*128)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def main():
    if args.resume:
        model.load_model()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        torch.save(model.state_dict(), '../models/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))
        model.save_model()


if __name__ == '__main__':
    main()