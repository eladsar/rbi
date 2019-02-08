import torch
from torch import nn
from preprocess import hinv_torch
from config import consts, args
import numpy as np


action_space = len(np.nonzero(consts.actions[args.game])[0])
batch_momentum = 0.001


class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        # value net
        self.fc_v = nn.Sequential(
            # nn.utils.weight_norm(nn.Linear(3136, args.hidden_features)),
            nn.Linear(3136, args.hidden_features),
            # nn.BatchNorm1d(args.hidden_features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
            # nn.utils.weight_norm(nn.Linear(args.hidden_features, 1)),
            nn.Linear(args.hidden_features, 1),
        )

        # advantage net
        self.fc_adv = nn.Sequential(
            # nn.utils.weight_norm(nn.Linear(3136, args.hidden_features)),
            nn.Linear(3136, args.hidden_features),
            # nn.BatchNorm1d(args.hidden_features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(args.hidden_features, action_space)),
            # nn.Linear(args.hidden_features, action_space),
        )

        # batch normalization and dropout
        self.cnn = nn.Sequential(
            # nn.utils.weight_norm(nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4)),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=batch_momentum, affine=False),
            nn.ReLU(),
            # nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=False),
            nn.ReLU(),
            # nn.utils.weight_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=False),
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

        return v, adv, adv_a, q, q_a


class BehavioralNet(nn.Module):

    def __init__(self):

        super(BehavioralNet, self).__init__()

        # batch normalization and dropout
        self.cnn = nn.Sequential(
            # nn.utils.weight_norm(nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4)),
            nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=batch_momentum, affine=False),
            nn.ReLU(),
            # nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=False),
            nn.ReLU(),
            # nn.utils.weight_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=batch_momentum, affine=False),
            nn.ReLU(),
        )

        # advantage net
        self.fc_beta = nn.Sequential(
            # nn.utils.weight_norm(nn.Linear(3136, args.hidden_features)),
            nn.Linear(3136, args.hidden_features),
            # nn.BatchNorm1d(args.hidden_features, eps=1e-05, momentum=batch_momentum, affine=True),
            nn.ReLU(),
            # nn.utils.weight_norm(nn.Linear(args.hidden_features, action_space)),
            nn.Linear(args.hidden_features, action_space),
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
        beta = self.fc_beta(s)

        return beta


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