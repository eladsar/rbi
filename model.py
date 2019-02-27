import torch
from torch import nn
from config import consts, args
import numpy as np


action_space = len(np.nonzero(consts.actions[args.game])[0])


class VarLayer(nn.Module):

    def __init__(self, size):
        super(VarLayer, self).__init__()
        self.generator = torch.distributions.normal.Normal(torch.zeros(size), torch.ones(size))

    def forward(self, batch, mean, logvar):

        if self.training:
            return mean + self.generator.sample((batch,)) * torch.exp(0.5 * logvar)
        return mean


class AutoEncoder(nn.Module):

    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.mean = nn.Linear(3136, args.hidden_features)
        self.std = nn.Linear(3136, args.hidden_features)
        self.var_layer = VarLayer(args.hidden_features)

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

        self.dconv = nn.Sequential(
            nn.Linear(args.hidden_features, 3136),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, args.history_length, kernel_size=8, stride=4),
        )

        # initialization
        self.dcnn[2].bias.data.zero_()
        self.dcnn[4].bias.data.zero_()
        self.dcnn[6].bias.data.zero_()

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s):

        # state CNN
        batch = s.size(0)
        s = self.cnn(s)
        s = s.view(batch, -1)

        mean = self.mean(s)
        logvar = self.logvar(s)

        s = self.var_layer(batch, mean, logvar)
        s = s.view(s.size(0), 64, 7, 7)
        s = self.cnn(s)

        return s, mean, torch.exp(0.5 * logvar)


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Linear(args.hidden_features, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            nn.Linear(args.hidden_features, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, action_space),
        )

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s, a, beta):

        v = self.fc_v(s)
        adv_tilde = self.fc_adv(s)

        bias = (adv_tilde * beta).sum(1).unsqueeze(1)
        adv = adv_tilde - bias

        adv_a = adv.gather(1, a).squeeze(1)
        q = v + adv

        q_a = q.gather(1, a).squeeze(1)

        return v, adv, adv_a, q, q_a


class BehavioralNet(nn.Module):

    def __init__(self):

        super(BehavioralNet, self).__init__()

        # advantage net
        self.fc_beta = nn.Sequential(
            nn.Linear(args.hidden_features, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, action_space),
        )

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, s):

        beta = self.fc_beta(s)

        return beta


class DuelRNN(nn.Module):

    def __init__(self):

        super(DuelRNN, self).__init__()

        self.hidden_rnn = int(args.hidden_features_rnn / 2)
        # self.hidden_rnn = int(args.hidden_features_rnn)

        # advantage net
        self.fc_adv = nn.Sequential(
            nn.Linear(self.hidden_rnn, args.hidden_features),
            # nn.Linear(3136, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, action_space),
        )

        self.fc_v = nn.Sequential(
            nn.Linear(self.hidden_rnn, args.hidden_features),
            # nn.Linear(3136, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, 1),
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

        # self.rnn = nn.LSTM(self.hidden_rnn, self.hidden_rnn, 1, batch_first=True, dropout=0, bidirectional=False)
        self.rnn = nn.LSTM(3136, self.hidden_rnn, 1, batch_first=True, dropout=0, bidirectional=False)

        # initialization
        self.cnn[0].bias.data.zero_()
        self.cnn[2].bias.data.zero_()
        self.cnn[4].bias.data.zero_()

    def forward(self, s, a, beta, h):

        # state CNN
        batch, seq, channel, height, width = s.shape
        s = s.view(batch * seq, channel, height, width)
        s = self.cnn(s)
        s = s.view(batch, seq, 3136)

        h.unsqueeze_(0)
        h = h.view(1, batch, self.hidden_rnn, 2)
        s, h = self.rnn(s, (h[:,:,:,0].contiguous(), h[:,:,:,1].contiguous()))
        h = torch.cat(h, dim=2)
        #
        # s, h = self.rnn(s, h)
        h.squeeze_(0)

        v = self.fc_v(s)
        adv_tilde = self.fc_adv(s)
        bias = (adv_tilde * beta).sum(2).unsqueeze(2)
        adv = adv_tilde - bias

        q = v + adv
        q_a = q.gather(2, a).squeeze(2)

        return q, q_a, h.detach()


class BehavioralRNN(nn.Module):

    def __init__(self):

        super(BehavioralRNN, self).__init__()

        self.hidden_rnn = int(args.hidden_features_rnn / 2)

        # batch normalization and dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(3136, self.hidden_rnn, 1, batch_first=True, dropout=0, bidirectional=False)
        # self.rnn = nn.GRU(3136, self.hidden_rnn, 1, batch_first=True, dropout=0, bidirectional=False)

        # behavior net
        self.fc_beta = nn.Sequential(
            # nn.Linear(3136, args.hidden_features),
            nn.Linear(self.hidden_rnn, args.hidden_features),
            nn.ReLU(),
            nn.Linear(args.hidden_features, action_space),
        )

        # initialization
        self.cnn[0].bias.data.zero_()
        self.cnn[2].bias.data.zero_()
        self.cnn[4].bias.data.zero_()

    def forward(self, s, h):

        # state CNN
        batch, seq, channel, height, width = s.shape
        s = s.view(-1, channel, height, width)

        # batch_seq, channel, height, width = s.shape
        s = self.cnn(s)
        s = s.view(batch, seq, 3136)

        h.unsqueeze_(0)
        h = h.view(1, batch, self.hidden_rnn, 2)
        s, h = self.rnn(s, (h[:,:,:,0].contiguous(), h[:,:,:,1].contiguous()))
        h = torch.cat(h, dim=2)
        h.squeeze_(0)

        beta = self.fc_beta(s)

        return beta, h.detach()
