import torch
from torch import nn
import torch.nn.functional as F
import math
from config import args, exp

if args.bias_relu:
    set_bias = True
    activation = torch.nn.ReLU
else:
    set_bias = False
    activation = torch.nn.LeakyReLU


class VarLayer(nn.Module):

    def __init__(self):
        super(VarLayer, self).__init__()

    def forward(self, mean, std):

        if self.training:

            if mean.is_cuda:
                return mean + torch.cuda.FloatTensor(std.shape).normal_() * std
            else:
                return mean + torch.FloatTensor(std.shape).normal_() * std

        return mean


class PolicyNet(nn.Module):

    def __init__(self):

        super(PolicyNet, self).__init__()

        # batch normalization and dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=set_bias),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=set_bias),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=set_bias),
            activation(),
        )

        # advantage net
        self.fc_pi = nn.Sequential(
            nn.Linear(3136, 512, bias=set_bias),
            activation(),
            nn.Linear(512, exp.labels_num, bias=set_bias),
        )

        if args.bias_relu:
            # initialization
            self.cnn[0].bias.data.zero_()
            self.cnn[2].bias.data.zero_()
            self.cnn[4].bias.data.zero_()

    def forward(self, s, stat=False):

        # state CNN

        s = self.cnn(s)
        s = s.view(s.size(0), -1)
        pi = self.fc_pi(s)

        if stat:
            return pi, {}

        return pi


class PredictNet(nn.Module):

    def __init__(self):

        super(PredictNet, self).__init__()

        self.latent_dim = args.latent
        self.labels_num = exp.labels_num

        self.mu = nn.Linear(3136, self.latent_dim * self.labels_num, bias=set_bias)
        self.rho = nn.Linear(3136, self.latent_dim * self.labels_num, bias=set_bias)
        self.var_layer = VarLayer()

        # batch normalization and dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=set_bias),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=set_bias),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=set_bias),
            activation(),
        )

        self.dlin = nn.Sequential(
            nn.Linear(self.latent_dim, 3136, bias=set_bias),
            activation(),
        )

        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, bias=set_bias),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=set_bias),
            activation(),
            nn.ConvTranspose2d(32, 4, kernel_size=8, stride=4, bias=set_bias),
        )

        if args.bias_relu:
            # initialization
            self.cnn[0].bias.data.zero_()
            self.cnn[2].bias.data.zero_()
            self.cnn[4].bias.data.zero_()

            # initialization
            self.dconv[0].bias.data.zero_()
            self.dconv[2].bias.data.zero_()
            self.dconv[4].bias.data.zero_()

        self.std_forget = 1
        self.mu_forget = -1

        self.std_learn = 1
        self.mu_learn = 1

        self.gen_mu = {'learn': self.mu_learn, 'forget': self.mu_forget, 'avg': -0.5, }
        self.gen_std = {'learn': self.std_learn, 'forget': self.std_forget, 'avg': 1, }

    def gen_samples(self, batch, source='learn'):

        mu = self.gen_mu[source]
        std = self.gen_std[source]

        mu = torch.cuda.FloatTensor(batch, self.latent_dim).fill_(mu)
        std = torch.cuda.FloatTensor(batch, self.latent_dim).fill_(std)

        # std = torch.abs(rho)
        z = self.var_layer(mu, std)

        # decode from latent variable y_hat
        x = self.dlin(z)
        x = x.view(batch, 64, 7, 7)
        x = self.dconv(x)

        return x

    def forward(self, s, a=None):

        # state CNN
        batch = s.size(0)
        s = self.cnn(s)
        s = s.view(batch, -1)

        mu = self.mu(s)
        rho = self.rho(s)
        std = torch.log1p(torch.exp(rho))

        if a is None:
            kl = (torch.log(self.std_forget / std) +
              (std ** 2 + (mu - self.mu_forget) ** 2) / (2 * self.std_forget ** 2) - 0.5).sum(dim=1)
            return mu, std, kl

        a = a.unsqueeze(1)
        a_latent = a.unsqueeze(2).repeat(1, self.latent_dim, 1)

        z = self.var_layer(mu, std)

        # attention
        z = z.view(-1, self.latent_dim, self.labels_num)
        z = z.gather(2, a_latent).squeeze(2)

        x = self.dlin(z)
        x = x.view(batch, 64, 7, 7)
        x = self.dconv(x)

        mu = mu.view(-1, self.latent_dim, self.labels_num)
        std = std.view(-1, self.latent_dim, self.labels_num)

        # calculate the kl loss
        kl_forget = (torch.log(self.std_forget / std) +
              (std ** 2 + (mu - self.mu_forget) ** 2) / (2 * self.std_forget ** 2) - 0.5).sum(dim=1)

        kl_learn = (torch.log(self.std_learn / std) +
              (std ** 2 + (mu - self.mu_learn) ** 2) / (2 * self.std_learn ** 2) - 0.5).sum(dim=1)

        kl_forget_y = kl_forget.gather(1, a).squeeze(1)
        kl_learn_y = kl_learn.gather(1, a).squeeze(1)

        kl = kl_forget.sum(dim=1) - kl_forget_y + kl_learn_y

        return x, mu, std, kl
