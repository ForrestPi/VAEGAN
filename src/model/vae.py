import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from model.modules import ConvBNLReLU, UpsampleNearestCBLR


class Encoder(nn.Module):
    def __init__(self, nc, fmaps, latent_variable_size):
        super().__init__()

        self.nc = nc
        self.fmaps = fmaps
        self.latent_variable_size = latent_variable_size

        self.layer1 = ConvBNLReLU(nc, fmaps, 4, 2, 1)
        self.layer2 = ConvBNLReLU(fmaps, fmaps * 2, 4, 2, 1)
        self.layer3 = ConvBNLReLU(fmaps * 2, fmaps * 4, 4, 2, 1)
        self.layer4 = ConvBNLReLU(fmaps * 4, fmaps * 8, 4, 2, 1)
        self.layer5 = ConvBNLReLU(fmaps * 8, fmaps * 8, 4, 2, 1)

        self.fc1 = nn.Linear(fmaps * 8 * 4 * 4, latent_variable_size)
        self.fc2 = nn.Linear(fmaps * 8 * 4 * 4, latent_variable_size)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        h5 = self.layer5(h4)
        h5 = h5.view(x.shape[0], -1)
        mu = self.fc1(h5)
        logvar = self.fc2(h5)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_variable_size, fmaps, nc, add_sigmoid=False):
        super().__init__()
        self.latent_variable_size = latent_variable_size
        self.fmaps = fmaps
        self.nc = nc

        self.fc = nn.Sequential(
            nn.Linear(latent_variable_size, fmaps*8*2*4*4),
            nn.ReLU()
        )
        tch = fmaps * 8 * 2
        self.layer5 = UpsampleNearestCBLR(tch, tch//2, 3, 1)
        tch //= 2
        self.layer4 = UpsampleNearestCBLR(tch, tch//2, 3, 1)
        tch //=2
        self.layer3 = UpsampleNearestCBLR(tch, tch // 2, 3, 1)
        tch //= 2
        self.layer2 = UpsampleNearestCBLR(tch, tch // 2, 3, 1)
        tch //= 2
        self.layer1 = UpsampleNearestCBLR(tch, tch // 2, 3, 1)
        tch //= 2
        self.out = nn.Sequential(
            nn.Conv2d(tch, nc, 1)
        )
        if add_sigmoid:
            self.out.add_module("sigmoid", nn.Sigmoid())

    def forward(self, z):
        h5 = self.fc(z)
        h5 = h5.view(z.shape[0], -1, 4, 4)
        h4 = self.layer5(h5)
        h3 = self.layer4(h4)
        h2 = self.layer3(h3)
        h1 = self.layer2(h2)
        x = self.layer1(h1)

        x = self.out(x)
        return x


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        latent_variable_size = args.latent_variable_size
        fmaps = args.fmaps
        nc = args.nc

        if args.rec_loss == "bce":
            add_sigmoid = True
        else:
            add_sigmoid = False

        self.nc = nc
        self.fmaps = fmaps
        self.latent_variable_size = latent_variable_size

        self.encode = Encoder(nc, fmaps, latent_variable_size)
        self.decode = Decoder(latent_variable_size, fmaps, nc, add_sigmoid)

    def reparametrize(self, mu, logvar):
        sigma = logvar.mul(0.5).exp()
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    def save_model(self, model_dir="../model", model_name="vae.pkl"):
        torch.save(self.state_dict(), os.path.join(model_dir, model_name))
        return

    def load_model(self, model_dir="../model", model_name="vae.pkl"):
        self.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        return