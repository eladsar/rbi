import torch
from torch import nn
from preprocess import hinv_torch
from config import consts, args
import numpy as np

drop_in = 0.25
drop_out = 0.1  # 0.5

drop_arg = args.dropout

# features = args.hidden_features
# if drop_arg:
#     features_behavior = int(args.hidden_features / drop_out)
# else:

features_behavior = args.hidden_features
features_value = args.hidden_features

aux_features = 1
batch_momentum = 0.0002
action_space = len(np.nonzero(consts.actions[args.game])[0])


class BehavioralNet(nn.Module):

    def __init__(self, drop=True):

        super(BehavioralNet, self).__init__()

        if drop and drop_arg:
            drop_out_this = drop_out
            drop_in_this = drop_in
            # drop_out_this = 0
            # drop_in_this = 0
        else:
            drop_out_this = 0
            drop_in_this = 0

        # policy estimator
        self.fc_beta = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(features_behavior, action_space),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout(drop_in_this),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            # nn.InstanceNorm2d(32, eps=1e-05),
            # nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, bias=False),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(64, eps=1e-05),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.InstanceNorm2d(64, eps=1e-05),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_h = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(3136 + aux_features, features_behavior),
            # nn.InstanceNorm1d(features, eps=1e-05),
            # nn.Linear(3136 + aux_features, features, bias=False),
            # nn.BatchNorm1d(features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def forward(self, s, aux):

        # state CNN

        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)
        # s = self.cnn_h(torch.cat((s.view(s.size(0), -1), aux), dim=1).unsqueeze(1)).squeeze(1)
        s = self.cnn_h(torch.cat((s.view(s.size(0), -1), aux), dim=1))
        # behavioral estimator
        beta = self.fc_beta(s)

        return beta


class DuelNet(nn.Module):

    def __init__(self, drop=True):

        super(DuelNet, self).__init__()

        if drop and drop_arg:
            drop_out_this = drop_out
            drop_in_this = drop_in
            # drop_out_this = 0
            # drop_in_this = 0
        else:
            drop_out_this = 0
            drop_in_this = 0

        # value net
        self.fc_v = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(features_value, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(features_value, action_space),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout(drop_in_this),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            # nn.InstanceNorm2d(32, eps=1e-05),
            # nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, bias=False),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(64, eps=1e-05),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.InstanceNorm2d(64, eps=1e-05),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
        )

        self.fc_h_v = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(3136 + aux_features, features_value),
            # nn.InstanceNorm1d(features, eps=1e-05),
            # nn.Linear(3136 + aux_features, features, bias=False),
            # nn.BatchNorm1d(features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
        )

        self.fc_h_a = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(3136 + aux_features, features_value),
            # nn.InstanceNorm1d(features, eps=1e-05),
            # nn.Linear(3136 + aux_features, features, bias=False),
            # nn.BatchNorm1d(features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def forward(self, s, a, beta, aux):

        # state CNN

        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)

        # phi = s.detach()
        # s = torch.cat((s.view(s.size(0), -1), aux), dim=1).unsqueeze(1)

        s = torch.cat((s.view(s.size(0), -1), aux), dim=1)
        s_v = self.fc_h_v(s)
        s_a = self.fc_h_a(s)

        # behavioral estimator
        v = self.fc_v(s_v)

        adv_tilde = self.fc_adv(s_a)

        bias = (adv_tilde * beta).sum(1).unsqueeze(1)
        # bias = adv_tilde.mean(1).unsqueeze(1)
        bias = bias.repeat(1, action_space)

        adv = adv_tilde - bias

        adv_a = adv.gather(1, a).squeeze(1)
        q = v.repeat(1, action_space) + adv
        # q = hinv_torch(q)
        q_a = q.gather(1, a).squeeze(1)

        return v, adv, adv_a, q, q_a


class ValueNet(nn.Module):

    def __init__(self, drop=True):

        super(ValueNet, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Linear(features_value, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            nn.Linear(features_value, action_space),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            # nn.Dropout(drop_in_this),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            # nn.InstanceNorm2d(32, eps=1e-05),
            # nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, bias=False),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(64, eps=1e-05),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.InstanceNorm2d(64, eps=1e-05),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
        )

        self.fc_h_v = nn.Sequential(
            # nn.Dropout(drop_out_this),
            nn.Linear(3136 + aux_features, features_value),
            # nn.InstanceNorm1d(features, eps=1e-05),
            # nn.Linear(3136 + aux_features, features, bias=False),
            # nn.BatchNorm1d(features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def forward(self, s, aux):

        # state CNN

        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)

        s = torch.cat((s.view(s.size(0), -1), aux), dim=1)
        s_v = self.fc_h_v(s)

        v = self.fc_v(s_v)

        return v


class DQN(nn.Module):

    def __init__(self, drop=True):

        super(DQN, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Linear(512, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            nn.Linear(512, action_space),
        )

        # batch normalization and dropout
        self.cnn_conv1 = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )

        self.cnn_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.cnn_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_h_v = nn.Sequential(
            nn.Linear(3136 + aux_features, 512),
            nn.ReLU(),
        )

        self.fc_h_a = nn.Sequential(
            nn.Linear(3136 + aux_features, 512),
            nn.ReLU(),
        )

        # initialization
        self.cnn_conv1[0].bias.data.zero_()
        self.cnn_conv2[0].bias.data.zero_()
        self.cnn_conv3[0].bias.data.zero_()

    def forward(self, s, a, aux):

        # state CNN

        s = self.cnn_conv1(s)
        s = self.cnn_conv2(s)
        s = self.cnn_conv3(s)

        s = torch.cat((s.view(s.size(0), -1), aux), dim=1)
        s_v = self.fc_h_v(s)
        s_a = self.fc_h_a(s)

        # behavioral estimator
        v = self.fc_v(s_v)

        adv_tilde = self.fc_adv(s_a)

        bias = adv_tilde.mean(1).unsqueeze(1)
        bias = bias.repeat(1, action_space)

        adv = adv_tilde - bias

        adv_a = adv.gather(1, a).squeeze(1)
        q = v.repeat(1, action_space) + adv
        q_a = q.gather(1, a).squeeze(1)

        return v, adv, adv_a, q, q_a


class DuelRNN(nn.Module):

    def __init__(self):

        super(DuelRNN, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            nn.Linear(args.hidden_features_rnn, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            nn.Linear(args.hidden_features_rnn, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
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

        self.pre_lstm = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU(),
        )

        self.rnn_adv = nn.GRU(128, args.hidden_features_rnn, 1, batch_first=True, dropout=0, bidirectional=False)
        self.rnn_v = nn.GRU(128, args.hidden_features_rnn, 1, batch_first=True, dropout=0, bidirectional=False)

        # initialization
        self.cnn[0].bias.data.zero_()
        self.cnn[2].bias.data.zero_()
        self.cnn[4].bias.data.zero_()

    def forward(self, s, a, beta, h_adv, h_v, batch):

        # state CNN
        # batch, seq, channel, height, width = s.shape
        # s = s.view(batch * seq, channel, height, width)
        # s = self.cnn(s)
        # s = s.view(batch, seq, 3136)

        # batch_seq, channel, height, width = s.shape
        s = self.cnn(s)
        s = s.view(batch, -1, 3136)

        s = self.pre_lstm(s)
        s = s.contiguous()

        s_adv, h_adv = self.rnn_adv(s, h_adv)
        s_v, h_v = self.rnn_v(s, h_v)

        v = self.fc_v(s_v)
        adv_tilde = self.fc_adv(s_adv)

        bias = (adv_tilde * beta).sum(2).unsqueeze(2)
        adv = adv_tilde - bias

        adv_a = adv.gather(2, a).squeeze(2)
        q = v + adv

        # q = hinv_torch(q)
        q_a = q.gather(2, a).squeeze(2)

        return v, adv, adv_a, q, q_a, h_adv.detach(), h_v.detach()


class BehavioralRNN(nn.Module):

    def __init__(self, drop=True):

        super(BehavioralRNN, self).__init__()

        self.batch = args.batch
        # batch normalization and dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.pre_lstm = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU(),
        )

        # advantage net
        self.fc_beta = nn.Sequential(
            nn.Linear(args.hidden_features_rnn, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
        )

        self.rnn = nn.GRU(128, args.hidden_features_rnn, 1, batch_first=True, dropout=0, bidirectional=False)

        # initialization
        self.cnn[0].bias.data.zero_()
        self.cnn[2].bias.data.zero_()
        self.cnn[4].bias.data.zero_()

    def forward(self, s, h, batch):

        # state CNN

        # state CNN
        # batch, seq, channel, height, width = s.shape
        # s = s.view(-1, channel, height, width)

        # batch_seq, channel, height, width = s.shape
        s = self.cnn(s)
        s = s.view(batch, -1, 3136)
        s = self.pre_lstm(s)
        s = s.contiguous()

        s, h = self.rnn(s, h)
        beta = self.fc_beta(s)

        return beta, h.detach()


class HInv(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        a = torch.abs(input)
        piece = (input <= 5).float()
        return torch.sign(input) * (2 * a * piece + (a ** 2 - 3 * a + 6.25) * (1 - piece))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output