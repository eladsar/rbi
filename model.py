import torch
from torch import nn
from config import consts, args
import numpy as np


action_space = len(np.nonzero(consts.actions[args.game])[0])


'''class GaussianProcess(torch.autograd.Function):

    prior_sigma = 0.1

    @staticmethod
    def forward(ctx, mean, logvar):
        x = torch.cuda.FloatTensor(mean.shape).normal_()
        ctx.save_for_backward(x, mean, logvar)
        return mean + x * torch.exp(0.5 * logvar)

    @staticmethod
    def backward(ctx, grad_output):

        scale = (GaussianProcess.prior_sigma ** 2)
        x, mean, logvar = ctx.saved_tensors
        grad_mean = grad_output + mean / scale

        var = torch.exp(logvar)
        grad_logvar = 0.5 * (grad_output * x * var ** 0.5 + var / scale - torch.cuda.FloatTensor(logvar.shape).fill_(1))

        return grad_mean, grad_logvar'''


class GaussianProcess(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mean, rho):
        x = torch.cuda.FloatTensor(mean.shape).normal_()
        ctx.save_for_backward(x, mean, rho)
        std = torch.log1p(torch.exp(rho))
        return mean + x * std

    @staticmethod
    def backward(ctx, grad_output):

        x, mean, rho = ctx.saved_tensors

        # grad_mean = grad_output
        # std_tag = torch.exp(rho) / (1 + torch.exp(rho))
        # grad_std = x * std_tag

        grad_mean = grad_output + mean

        std = torch.log1p(torch.exp(rho))
        std_tag = torch.exp(rho) / (1 + torch.exp(rho))
        grad_std = x * std_tag + std_tag * std - std_tag / std

        return grad_mean, grad_std


gaussian_process = GaussianProcess.apply


class StochsticLayer(nn.Module):

    def __init__(self, deterministic_layer, *args, **kwargs):
        super(StochsticLayer, self).__init__()
        self.mean = deterministic_layer(*args, **kwargs)
        self.logvar = deterministic_layer(*args, **kwargs)

    def forward(self, x):
        mean = self.mean(x)

        if self.training:
            logvar = self.logvar(x)
            y = gaussian_process(mean, logvar)
        else:
            y = mean

        return y


class StochsticWeight(nn.Module):

    def __init__(self, deterministic_function, weight, bias, *args, **kwargs):
        super(StochsticWeight, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.function = deterministic_function

        self.weight_mean = torch.cuda.FloatTensor(weight, requires_grad=True)
        self.weight_logvar = torch.cuda.FloatTensor(weight, requires_grad=True)

        if bias is not None:
            self.bias_mean = torch.cuda.FloatTensor(bias, requires_grad=True)
            self.bias_logvar = torch.cuda.FloatTensor(bias, requires_grad=True)
        else:
            self.bias_mean = None
            self.bias_logvar = None

    def forward(self, x):

        if self.training:
            weight = gaussian_process(self.weight_mean, self.weight_logvar)
            if self.bias_mean is not None:
                bias = gaussian_process(self.bias_mean, self.bias_logvar)
            else:
                bias = None

        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return self.function(x, weight, bias=bias, *self.args, **self.kwargs)


class Encoder(nn.Module):

    def __init__(self, stochastic=True):

        super(Encoder, self).__init__()

        if stochastic:
            self.latent_layer = StochsticLayer(nn.Linear, 3136, args.hidden_features, bias=False)
        else:
            self.latent_layer = nn.Linear(3136, args.hidden_features)

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

    def forward(self, s):

        # state CNN
        s = self.cnn(s)
        s = s.view(s.size(0), -1)
        s = self.latent_layer(s)

        return s


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        self.encoder = Encoder(stochastic=True)

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

        s = self.encoder(s)

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

        self.encoder = Encoder(stochastic=True)

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

        s = self.encoder(s)
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
