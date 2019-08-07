from model_stm import PolicyNet, PredictNet, PolicyNet2
import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

from config import consts, args
from torch.optim import Optimizer

import cv2
import os
import time
import math
import itertools


class PASGD(Optimizer):

    def __init__(self, params, lr=None, momentum=0., dampening=0.,
                 weight_decay=0., friction=0.1, nesterov=False, punctuation=1., warm_start=1000):

        self.theta_dist = 0
        self.i = 0

        if lr is None or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        friction=friction, punctuation=punctuation,
                        warm_start=warm_start)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PASGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PASGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def snapshot(self):

        self.i += 1
        for group in self.param_groups:
            # punctuation = group['punctuation']
            for p in group['params']:

                state = self.state[p]
                if 'theta_0' in state:
                    # p.data.add_(-friction*lr, (p.data - state['theta_0']))
                    self.calc_dist()
                if 'momentum_buffer' in state:
                    state.pop('momentum_buffer')

                state['theta_0'] = p.data.clone()

                # state['mask'] = torch.cuda.FloatTensor(p.shape[0]).bernoulli_(p=punctuation).unsqueeze(1).repeat(1, p.shape[1])

    def calc_dist(self):

        dist = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                dist += ((state['theta_0'] - p.data) ** 2).sum()

        self.theta_dist = dist

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            friction = group['friction']
            warm_start = group['warm_start']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p * param_state['mask'])
                p.data.add_(-group['lr'], d_p)

                if self.i >= warm_start:
                    p.data.add_(-friction, (p.data - param_state['theta_0']))

        return loss


def derangement(n):

    x = torch.randperm(n)

    y = torch.arange(n)

    loc = y[(x == y)]
    # nonloc = y[(x != y)][:len(loc)]
    nonloc = y[(x != y)]
    nonloc = nonloc[torch.randperm(len(nonloc))[:len(loc)]]

    g1 = x[loc]
    g2 = x[nonloc]

    x[loc] = g2
    x[nonloc] = g1

    return x


def gen_fake(x1, x2):

    x2 = x2[derangement(len(x2))]

    # z1 = 0.5 * x1 + 0.5 * x2
    z2 = torch.max(x1, x2)
    # z3 = torch.min(x1, x2)

    x1t = x1.transpose(0, -1)
    x2t = x2.transpose(0, -1)
    n = int(len(x1t) / 2)
    z4 = torch.cat([x1t[:n], x2t[n:]]).transpose(0, -1)

    x1t = x1.transpose(0, -2)
    x2t = x2.transpose(0, -2)
    n = int(len(x1t) / 2)
    z5 = torch.cat([x1t[:n], x2t[n:]]).transpose(0, -2)

    z = torch.cat([z2, z4, z5])
    idx = torch.randperm(len(z))[:len(x1)]
    z = z[idx]

    return z


def gen_fake_latent(z1, z2=None, mask=None, mask_type='punct', mask2=None):

    if z2 is None:
        z2 = z1
        factor = 2
    else:
        factor = 1

    z2 = z2[derangement(len(z2))]

    if mask is None:
        mask = torch.zeros_like(z2).bernoulli_(p=0.5)

    if mask_type is 'const':
        ind = torch.arange(args.latent)
        mask = (torch.stack([ind % 2, (ind + 1) % 2])).repeat(int(len(z1)/2), 1)
        mask = mask.float().to(exp.device)

    z = z1 * mask + z1 * (1 - mask) / 2
    zmu = z1.mean(dim=0).unsqueeze(0).repeat(len(z1), 1)

    z = z * mask2 + zmu * (1 - mask2)

    idx = torch.randperm(len(z))[:int(len(z1) / factor)]
    z = z[idx]

    return z


class Fifo(object):

    def __init__(self, length, size):
        self.length = length
        self.size = size
        self.x = torch.cuda.FloatTensor([])
        self.y = torch.cuda.LongTensor([])

    def push(self, x, y):

        if len(self.x) == 2 * self.size:
            sample = {'x_keep': self.x[-self.size:],
                      'y_keep': self.y[-self.size:],
                      'x_forget': self.x[:self.size],
                      'y_forget': self.y[:self.size],
                     }
        else:
            sample = None

        self.x = torch.cat([self.x, x[::self.length]])
        self.y = torch.cat([self.y, y[::self.length]])

        self.x = self.x[-(2 * self.size):]
        self.y = self.y[-(2 * self.size):]

        return sample

    def push2(self, x, y):

        if len(self.x) == self.size:
            sample = {'x_keep': self.x,
                      'y_keep': self.y,}
        else:
            sample = None

        self.x = torch.cat([self.x, x[::self.length]])
        self.y = torch.cat([self.y, y[::self.length]])

        self.x = self.x[-self.size:]
        self.y = self.y[-self.size:]

        return sample

    def reset(self):
        self.x = torch.cuda.FloatTensor([])
        self.y = torch.cuda.LongTensor([])


class STM(object):

    def __init__(self):

        self.networks_dict = {}
        self.optimizers_dict = {}

        # stm memory stuff
        self.latent_dim = args.latent
        self.labels_num = len(np.nonzero(consts.actions[args.game])[0])

        self.add_labels = 1
        self.net = PolicyNet2(add=1)

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        self.device = torch.device("cuda:%d" % args.cuda_default)
        self.net.to(self.device)
        self.net.train()

        self.optimizer = PASGD(self.net.parameters(), lr=0.00025/4, momentum=0.9, weight_decay=0,
                               nesterov=True, friction=0., punctuation=1,
                               warm_start=args.warm_start)

        self.loss_classification = nn.CrossEntropyLoss(reduction='none')

        self.mu_base_net = 10
        self.lagrange_base_net = 1
        self.batch = args.stm_batch

        self.learn_queue = torch.cuda.FloatTensor([])
        self.forget_queue = torch.cuda.FloatTensor([])
        self.action_queue = torch.cuda.LongTensor([])
        self.action_forget_queue = torch.cuda.LongTensor([])

        self.beta_net = 1
        self.th = -math.log(args.pa_threshold)
        self.disc_th = 0.8

        self.n_iter = args.n_iter
        self.current_n_iter = 0
        self.max_n_iter = args.max_iter

        self.fifo_learn = Fifo(args.fifo_size, self.batch)
        self.fifo_forget = Fifo(args.fifo_size, self.batch)

        self.statistics = {}

    def get_optimizers(self):

        self.optimizers_dict = {}

        for d in dir(self):
            x = getattr(self, d)
            if issubclass(type(x), torch.optim.Optimizer) and hasattr(x, 'state_dict'):
                self.optimizers_dict[d] = x

        return self.optimizers_dict

    def get_networks(self):

        self.networks_dict = {}
        name_dict = {}

        for d in dir(self):
            x = getattr(self, d)
            if issubclass(type(x), nn.Module) and hasattr(x, 'state_dict'):
                name_dict[d] = getattr(x, 'named_parameters')
                self.networks_dict[d] = x

        return name_dict

    def save_checkpoint(self, path=None, aux=None):

        if not self.networks_dict:
            self.get_networks()
        if not self.optimizers_dict:
            self.get_optimizers()

        state = {'aux': aux}

        for net in self.networks_dict:
            state[net] = copy.deepcopy(self.networks_dict[net].state_dict())

        for optimizer in self.optimizers_dict:
            state[optimizer] = copy.deepcopy(self.optimizers_dict[optimizer].state_dict())

        if path is not None:
            torch.save(state, path)

        return state

    def load_checkpoint(self, pathstate):

        if not self.networks_dict:
            self.get_networks()
            self.get_optimizers()

        if type(pathstate) is str:
            state = torch.load(pathstate, map_location="cuda:%d" % args.cuda_default)
        else:
            state = pathstate

        for net in self.networks_dict:
             self.networks_dict[net].load_state_dict(state[net])

        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].load_state_dict(state[optimizer])

        return state['aux']

    def fetch_gate(self, s):

        self.net.eval()

        a_hat, _, _, _ = self.net(s)
        a_hat = torch.softmax(a_hat, dim=1).detach()

        a_prob, a_ind = torch.max(a_hat[:, :self.labels_num], dim=1)
        disc = (a_prob > self.disc_th).float()
        prob = a_prob * disc

        return a_ind, prob

    def load_stat(self):
        if self.statistics:
            x = self.statistics
            self.statistics = {}
            return x
        else:
            return None

    def add_batch(self, s, a, target='learn'):

        if target == 'learn':
            self.learn_queue = torch.cat([self.learn_queue, s])[-self.batch:]
            self.action_queue = torch.cat([self.action_queue, a])[-self.batch:]

            if len(self.learn_queue) >= self.batch and len(self.forget_queue) >= self.batch:
                sample_learn = {'x_train': self.learn_queue, 'y_train': self.action_queue}
                sample_forget = {'x_forget': self.forget_queue, 'y_forget': self.action_forget_queue}
                self.learn(sample_learn, sample_forget)

                self.learn_queue = torch.cuda.FloatTensor([])
                self.forget_queue = torch.cuda.FloatTensor([])
                self.action_queue = torch.cuda.LongTensor([])
                self.action_forget_queue = torch.cuda.LongTensor([])

        elif target == 'forget':
            self.forget_queue = torch.cat([self.forget_queue, s])[-self.batch:]
            self.action_forget_queue = torch.cat([self.action_forget_queue, a])[-self.batch:]

        else:
            raise NotImplementedError

    def learn(self, sample_learn, sample_forget):

        # Policy part
        x_train = sample_learn['x_train'].to(self.device)
        y_train = sample_learn['y_train'].to(self.device)

        batch = len(x_train)

        sample_learn = self.fifo_learn.push2(x_train, y_train)

        x_forget = sample_forget['x_forget'].to(self.device)
        y_forget = sample_forget['y_forget'].to(self.device)
        sample_forget = self.fifo_forget.push2(x_forget, y_forget)

        if sample_learn is None:
            return

        self.net.train()
        self.statistics = {}
        self.optimizer.snapshot()

        x_keep = sample_learn['x_keep']
        y_keep = sample_learn['y_keep']

        x_forget2 = sample_forget['x_keep']

        y_forget = torch.cuda.LongTensor(len(x_forget)).fill_(self.labels_num)
        y_forget2 = torch.cuda.LongTensor(len(x_forget2)).fill_(self.labels_num)

        x_train = torch.cat([x_train, x_keep])
        y_train = torch.cat([y_train, y_keep])

        x_forget = torch.cat([x_forget, x_forget2])
        y_forget = torch.cat([y_forget, y_forget2])

        n_objectives = len(x_train) + len(x_forget)
        lagrange_net = torch.cuda.FloatTensor(n_objectives).fill_(self.lagrange_base_net)
        mu_net = torch.cuda.FloatTensor(n_objectives).fill_(self.mu_base_net)

        for n_iter in itertools.count(1):

            y_train_est, mu_train, std_train, kl_train = self.net(x_train)
            objective = self.loss_classification(y_train_est, y_train)

            y_forget_est, mu_forget, std_forget, kl_forget = self.net(x_forget)
            objective_forget = self.loss_classification(y_forget_est, y_forget)

            kl = torch.cat([kl_train, kl_forget])
            objective = torch.cat([objective, objective_forget])

            f_min = self.beta_net * kl.sum()

            loss_classification = f_min + (0.5 * mu_net * objective ** 2 + lagrange_net * objective).sum()
            loss_classification /= batch

            if n_iter >= self.max_n_iter or ((objective.max() <= self.th) and n_iter > self.n_iter):
                break

            self.optimizer.zero_grad()
            loss_classification.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 40, norm_type=2)
            self.optimizer.step()

            lagrange_net += mu_net * objective.detach()

        y_train_est = torch.softmax(y_train_est, dim=1)
        y_forget_est = torch.softmax(y_forget_est, dim=1)

        self.statistics['lagrange_net'] = float(lagrange_net[0])
        self.statistics['loss'] = float(loss_classification)
        self.statistics['theta_dist_net'] = float(self.optimizer.theta_dist)
        self.statistics['acc'] = float((y_train == y_train_est[:, :self.labels_num].argmax(dim=1)).float().mean())
        self.statistics['disc_t'] = float(y_train_est[:, self.labels_num].mean())
        self.statistics['disc_f'] = float(y_forget_est[:, self.labels_num].mean())
        self.statistics['n_iter'] = n_iter
        self.statistics['mu_train'] = float(mu_train.norm(2, dim=1).mean())
        self.statistics['std_train'] = float(std_train.mean())
        self.statistics['mu_forget'] = float(mu_forget.norm(2, dim=1).mean())
        self.statistics['std_forget'] = float(std_forget.mean())


    # def learn(self, sample_learn, sample_forget):
    #
    #     # Policy part
    #     x_train = sample_learn['x_train'].to(self.device)
    #     y_train = sample_learn['y_train'].to(self.device)
    #
    #     batch = len(x_train)
    #
    #     sample_learn = self.fifo_learn.push2(x_train, y_train)
    #
    #     x_forget = sample_forget['x_forget'].to(self.device)
    #     y_forget = sample_forget['y_forget'].to(self.device)
    #     sample_forget = self.fifo_forget.push2(x_forget, y_forget)
    #
    #     if sample_learn is None:
    #         return
    #
    #     self.net.train()
    #     self.statistics = {}
    #     self.optimizer.snapshot()
    #
    #     x_keep = sample_learn['x_keep']
    #     y_keep = sample_learn['y_keep']
    #
    #     x_forget = sample_forget['x_keep']
    #
    #     x_fake2 = gen_fake(x_keep, x_forget)
    #     mask = torch.cuda.FloatTensor(len(x_keep) * 2, self.latent_dim).bernoulli_(p=0.5)
    #     mask2 = torch.cuda.FloatTensor(len(x_keep) * 2, self.latent_dim).bernoulli_(p=0.75)
    #
    #     y_forget = torch.cuda.LongTensor(len(x_forget)).fill_(self.labels_num)
    #     y_fake = torch.cuda.LongTensor(len(x_keep)).fill_(self.labels_num)
    #     y_fake2 = torch.cuda.LongTensor(len(x_fake2)).fill_(self.labels_num)
    #
    #     x_train = torch.cat([x_train, x_keep, x_fake2])
    #     y_train = torch.cat([y_train, y_keep, y_fake2])
    #
    #     x_fake = torch.cat([x_forget, x_keep])
    #
    #     x_forget = torch.cat([x_forget])
    #     y_forget = torch.cat([y_forget])
    #
    #     n_objectives = len(x_train)
    #     lagrange_net = torch.cuda.FloatTensor(n_objectives).fill_(self.lagrange_base_net)
    #     mu_net = torch.cuda.FloatTensor(n_objectives).fill_(self.mu_base_net)
    #
    #     for n_iter in itertools.count(1):
    #
    #         y_train_est, _, _, kl_train = self.net(x_train)
    #         objective = self.loss_classification(y_train_est, y_train)
    #
    #         y_forget_est, _, _, kl_forget = self.net(x_forget)
    #         objective_forget = self.loss_classification(y_forget_est, y_forget)
    #
    #         z_fake, _, _, _ = self.net(x_fake, part='enc')
    #         z_fake = z_fake.detach()
    #         z_fake = gen_fake_latent(z_fake, mask=mask, mask2=mask2)
    #         y_fake_est = self.net(z_fake, part='dec')
    #         objective_fake = self.loss_classification(y_fake_est, y_fake)
    #
    #         kl = torch.cat([kl_train, kl_forget])
    #         objective_forget = torch.cat([objective_forget, objective_fake])
    #
    #         f_min = self.beta_net * kl.sum() + objective_forget.sum()
    #
    #         loss_classification = f_min + (0.5 * mu_net * objective ** 2 + lagrange_net * objective).sum()
    #         loss_classification /= batch
    #
    #         if n_iter >= self.max_n_iter or ((objective.max() <= self.th) and n_iter > self.n_iter):
    #             break
    #
    #         self.optimizer.zero_grad()
    #         loss_classification.backward()
    #         torch.nn.utils.clip_grad_norm_(self.net.parameters(), 40, norm_type=2)
    #         self.optimizer.step()
    #
    #         lagrange_net += mu_net * objective.detach()
    #
    #     acc = (y_train == y_train_est[:, :self.labels_num].argmax(dim=1)).mean()
    #     self.statistics['lagrange_net'] = float(lagrange_net[0])
    #     self.statistics['loss'] = float(loss_classification)
    #     self.statistics['theta_dist_net'] = float(self.optimizer.theta_dist)
    #     self.statistics['acc_policy'] = acc
    #     self.statistics['n_iter'] = n_iter


