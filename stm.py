from model_stm import PolicyNet, PredictNet
import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

from config import consts, args

import cv2
import os
import time
import math


class STM(object):

    def __init__(self):

        self.networks_dict = {}
        self.optimizers_dict = {}

        # stm memory stuff
        self.policy = PolicyNet()
        self.discriminator = PredictNet()

        if torch.cuda.device_count() > 1:
            self.policy = nn.DataParallel(self.policy)
            self.discriminator = nn.DataParallel(self.discriminator)

        self.device = torch.device("cuda:%d" % args.cuda_default)
        self.policy.to(self.device)
        self.policy.train()

        self.discriminator.to(self.device)
        self.discriminator.train()

        self.optimizer_net = torch.optim.Adam(self.policy.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)

        self.loss_classification = nn.CrossEntropyLoss()
        self.loss_discrimination = nn.BCEWithLogitsLoss()

        # parameters
        self.min_n_iter = 20
        self.max_n_iter = 200
        # self.real_weight = 0.1
        # self.fake_weight = 1 - self.real_weight
        self.min_acc = 0.99
        self.threshold_disc = 0.8
        self.threshold_policy = 0.9
        self.epoch_length = 128

        self.batch = args.stm_batch
        self.beta_prior = 0.1
        self.mu_disc = 0.1
        self.mu_net = 0.1

        self.learn_queue = torch.cuda.FloatTensor([])
        self.forget_queue = torch.cuda.FloatTensor([])
        self.action_queue = torch.cuda.LongTensor([])
        self.action_forget_queue = torch.cuda.LongTensor([])

        # punctuation
        self.punctuation = 1 - math.e ** (-5 / self.epoch_length)

        self.policy_s0 = list(map(lambda x: x.clone().detach(), self.policy.parameters()))
        self.policy_masks = list(map(lambda x: torch.cuda.FloatTensor(x.shape), self.policy.parameters()))
        # self.policy_punct_list = list(map(lambda x: int(0), self.policy.named_parameters()))
        self.policy_punct_list = list(self.policy.named_parameters())

        self.disc_s0 = list(map(lambda x: x.clone().detach(), self.discriminator.parameters()))
        self.disc_masks = list(map(lambda x: torch.cuda.FloatTensor(x.shape), self.discriminator.parameters()))
        # self.disc_punct_list = list(map(lambda x: int('mu' in x[0] or 'rho' in x[0]), self.discriminator.named_parameters()))
        self.disc_punct_list = list(self.discriminator.named_parameters())

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

    def fifo_add_batch(self, s, a, target='learn'):

        if target == 'learn':
            self.learn_queue = torch.cat([self.learn_queue, s])[-self.batch:]
            self.action_queue = torch.cat([self.action_queue, a])[-self.batch:]
        elif target == 'forget':
            self.forget_queue = torch.cat([self.forget_queue, s])[-self.batch:]
            self.action_forget_queue = torch.cat([self.action_forget_queue, a])[-self.batch:]

            if len(self.learn_queue) >= self.batch and len(self.forget_queue) >= self.batch:
                self.learn_routine(self.learn_queue, self.action_queue, self.forget_queue, self.action_forget_queue)

                self.learn_queue = torch.cuda.FloatTensor([])
                self.forget_queue = torch.cuda.FloatTensor([])
                self.action_queue = torch.cuda.LongTensor([])
                self.action_forget_queue = torch.cuda.LongTensor([])

            # elif len(self.learn_queue) >= self.batch:
            #
            #     self.learn_routine(self.learn_queue, self.action_queue)
            #     self.forget_queue = torch.cuda.FloatTensor([])
            #     self.action_forget_queue = torch.cuda.LongTensor([])



        else:
            raise NotImplementedError

    def fetch_gate(self, s):

        self.policy.train()
        self.discriminator.train()

        a_hat = self.policy(s)
        a_hat = a_hat.detach()
        a_prob, a_ind = torch.max(torch.softmax(a_hat, dim=1), dim=1)

        d, _, _, _ = self.discriminator(s, a_ind)
        z = (torch.sigmoid(d.squeeze(1)) > self.threshold_disc).long() * (a_prob > self.threshold_policy).long()

        return a_ind * z - (1 - z)

    def load_stat(self):
        if self.statistics:
            x = self.statistics
            self.statistics = {}
            return x
        else:
            return None

    def learn_routine(self, s_forget, a_forget, s_learn=None, a_learn=None):

        self.policy.train()
        self.discriminator.train()

        theta_snap_net = list(map(lambda x: x.clone().detach(), self.policy.parameters()))
        theta_snap_disc = list(map(lambda x: x.clone().detach(), self.discriminator.parameters()))

        # initialize augmented lagrange parameter
        lagrange_net = 1.
        lagrange_disc = 1.

        acc_policy = 0
        acc_discriminator = 0

        self.statistics = {}

        # get punctuation pattern
        masks_policy = list(map(lambda x: x[0].bernoulli_(p=self.punctuation if x[1] else 0),
                         zip(self.policy_masks, self.policy_punct_list)))

        with torch.no_grad():
            for param, mask, org in zip(self.policy.parameters(), masks_policy, self.policy_s0):
                param.data = mask * org.data + (1 - mask) * param

        # get punctuation pattern
        masks_disc = list(map(lambda x: x[0].bernoulli_(p=self.punctuation if x[1] else 0),
                         zip(self.disc_masks, self.disc_punct_list)))

        with torch.no_grad():
            for param, mask, org in zip(self.discriminator.parameters(), masks_disc, self.disc_s0):
                param.data = mask * org.data + (1 - mask) * param

        # Policy part

        if s_learn is not None:

            for k_net in range(self.max_n_iter):

                theta_current_net = list(self.policy.parameters())

                theta_dist_net = 0.5 * torch.stack([((1 - mask) * (theta_i - theta_j) ** 2).sum() for theta_i, theta_j, mask
                                                in zip(theta_snap_net, theta_current_net, masks_policy)]).sum()

                a_est = self.policy(s_learn)
                objective = self.loss_classification(a_est, a_learn)

                # PA
                loss_classification = theta_dist_net + self.mu_net / 2 * objective ** 2 + lagrange_net * objective

                # loss_classification = theta_dist_net + objective

                self.optimizer_net.zero_grad()
                loss_classification.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40, norm_type=2)
                self.optimizer_net.step()

                lagrange_net += self.mu_net * objective.detach()

                acc_policy = (a_learn == a_est.argmax(dim=1)).float().mean()

                if acc_policy > self.min_acc and k_net >= self.min_n_iter:
                    break

            self.statistics['lagrange_net'] = float(lagrange_net)
            self.statistics['loss_classification'] = float(loss_classification)
            self.statistics['theta_dist_net'] = float(theta_dist_net)
            self.statistics['acc_policy'] = acc_policy
            self.statistics['k_net'] = k_net

        # Discriminator part

        s_learn_len = len(s_learn) if s_learn is not None else 0
        real = torch.cuda.FloatTensor(s_learn_len, 1).fill_(1)
        fake = torch.cuda.FloatTensor(len(s_forget), 1).fill_(0)

        for k_disc in range(self.max_n_iter):

            theta_current_disc = list(self.discriminator.parameters())
            theta_dist_disc = 0.5 * torch.stack([((theta_i - theta_j) ** 2).sum() for theta_i, theta_j in zip(theta_snap_disc, theta_current_disc)]).sum()

            if s_learn is not None:
                d_real, _, _, kl = self.discriminator(s_learn, a_learn)
                real_objective = self.loss_discrimination(d_real, real)
                prior_objective = kl.mean()
            else:
                real_objective = 0
                prior_objective = 0

            d_fake, _, _, kl_forget = self.discriminator(s_forget, a_forget)
            fake_objective = self.loss_discrimination(d_fake, fake)
            prior_fake_objective = kl_forget.mean()

            objective = real_objective + fake_objective

            f_min = theta_dist_disc + (prior_objective + prior_fake_objective) * self.beta_prior
            # PA
            loss_discrimination = f_min + self.mu_disc / 2 * objective ** 2 + lagrange_disc * objective
            # loss_discrimination = theta_dist_disc + objective

            self.optimizer_disc.zero_grad()
            loss_discrimination.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 40, norm_type=2)
            self.optimizer_disc.step()

            lagrange_disc += self.mu_disc * objective.detach()

            d = torch.sigmoid(torch.cat([d_real, -d_fake]))
            acc_discriminator = d.float().mean()

            if acc_discriminator > self.min_acc and k_disc >= self.min_n_iter:
                break

        self.statistics['lagrange_disc'] = float(lagrange_disc)
        self.statistics['loss_discrimination'] = float(loss_discrimination)
        self.statistics['theta_dist_disc'] = float(theta_dist_disc)
        self.statistics['acc_disc'] = acc_discriminator
        self.statistics['k_disc'] = k_disc


