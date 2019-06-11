import torch
from torch import nn
import torch.nn.functional as F
import math
from config import consts, args
import numpy as np

# if args.bias_relu:
#     set_bias = True
#     activation = torch.nn.ReLU
# else:
set_bias = False
activation = torch.nn.LeakyReLU

action_space = len(np.nonzero(consts.actions[args.game])[0])


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Linear(3136, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            nn.Linear(3136, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, action_space),
        )

        # batch normalization and dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # initialization
        self.cnn[0].bias.data.zero_()
        self.cnn[2].bias.data.zero_()
        self.cnn[4].bias.data.zero_()

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s, a, beta):

        # state CNN
        s = self.cnn(s)
        s = s.view(s.size(0), -1)

        v = self.fc_v(s)
        adv_tilde = self.fc_adv(s)

        bias = (adv_tilde * beta).sum(1).unsqueeze(1)
        adv = adv_tilde - bias

        adv_a = adv.gather(1, a).squeeze(1)
        q = v + adv

        q_a = q.gather(1, a).squeeze(1)

        return v, adv, adv_a, q, q_a, s


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

        # advantage net
        self.fc_pi = nn.Sequential(
            nn.Linear(3136, 512, bias=set_bias),
            activation(),
            nn.Linear(512, 128, bias=set_bias),
            activation(),
            nn.Linear(128, action_space, bias=set_bias),
        )

    def forward(self, s):

        pi = self.fc_pi(s)

        return pi


class PredictNet(nn.Module):

    def __init__(self):

        super(PredictNet, self).__init__()

        self.latent_dim = args.latent
        self.labels_num = action_space

        self.encoder = nn.Sequential(
            nn.Linear(3136, 512, bias=set_bias),
            activation(),
            nn.Linear(512, 512, bias=set_bias),
        )

        self.mu = nn.Linear(512, self.latent_dim * self.labels_num, bias=set_bias)
        self.rho = nn.Linear(512, self.latent_dim * self.labels_num, bias=set_bias)
        self.var_layer = VarLayer()

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512, bias=set_bias),
            activation(),
            nn.Linear(512, 512, bias=set_bias),
            activation(),
            nn.Linear(512, 3136, bias=set_bias),
        )

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

        z = self.var_layer(mu, std)

        # decode from latent variable y_hat
        x = self.decoder(z)

        return x

    def forward(self, s, a=None):

        s_shape = s.shape
        s = self.encoder(s)

        mu = self.mu(s)
        rho = self.rho(s)
        std = torch.log1p(torch.exp(rho))

        if a is None:
            kl = (torch.log(self.std_forget / std) +
              (std ** 2 + (mu - self.mu_forget) ** 2) / (2 * self.std_forget ** 2) - 0.5).sum(dim=1)

            mu = mu.view(-1, self.latent_dim, self.labels_num)
            std = std.view(-1, self.latent_dim, self.labels_num)

            return mu, std, kl

        a = a.unsqueeze(1)
        a_latent = a.unsqueeze(2).repeat(1, self.latent_dim, 1)

        z = self.var_layer(mu, std)

        # attention
        z = z.view(-1, self.latent_dim, self.labels_num)
        z = z.gather(2, a_latent).squeeze(2)

        x = self.decoder(z)

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

        x.view(s_shape)

        return x, mu, std, kl
