from model_stm import PolicyNet, PredictNet
import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import consts, args

import cv2
import os
import time
import math


class STM(object):

    def __init__(self):

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
        self.min_acc = 0.95

        self.batch = args.stm_batch

        self.learn_queue = torch.cuda.FloatTensor([])
        self.forget_queue = torch.cuda.FloatTensor([])
        self.action_queue = torch.cuda.LongTensor([])
        self.action_forget_queue = torch.cuda.LongTensor([])

        # punctuation
        self.punctuation = 1 - math.e ** (-5 / (128 * 10))

        self.policy_s0 = list(map(lambda x: x.clone().detach(), self.policy.parameters()))
        self.policy_masks = list(map(lambda x: torch.cuda.FloatTensor(x.shape), self.policy.parameters()))
        # self.policy_punct_list = list(map(lambda x: int(0), self.policy.named_parameters()))
        self.policy_punct_list = list(self.policy.named_parameters())

        self.disc_s0 = list(map(lambda x: x.clone().detach(), self.discriminator.parameters()))
        self.disc_masks = list(map(lambda x: torch.cuda.FloatTensor(x.shape), self.discriminator.parameters()))
        # self.disc_punct_list = list(map(lambda x: int('mu' in x[0] or 'rho' in x[0]), self.discriminator.named_parameters()))
        self.disc_punct_list = list(self.discriminator.named_parameters())

        self.statistics = {}

    def fifo_add_batch(self, s, a, target='learn'):

        if target == 'learn':
            self.learn_queue = torch.cat([self.learn_queue, s])[-self.batch:]
            self.action_queue = torch.cat([self.action_queue, a])[-self.batch:]
        elif target == 'forget':
            self.forget_queue = torch.cat([self.forget_queue, s])[-self.batch:]
            self.action_forget_queue = torch.cat([self.action_forget_queue, a])[-self.batch:]

        else:
            raise NotImplementedError

        if len(self.learn_queue) >= self.batch and len(self.forget_queue) >= self.batch:

            self.learn_routine(self.learn_queue, self.action_queue, self.forget_queue, self.action_forget_queue)

            self.learn_queue = torch.cuda.FloatTensor([])
            self.forget_queue = torch.cuda.FloatTensor([])
            self.action_queue = torch.cuda.LongTensor([])
            self.action_forget_queue = torch.cuda.LongTensor([])

    def fetch_gate(self, s):

        self.policy.train()
        self.discriminator.train()

        a_hat = self.policy(s)
        a_hat = a_hat.detach()
        a_prob, a_ind = torch.max(a_hat, dim=1)

        d, _, _, _ = self.discriminator(s, a_ind)
        z = (torch.sigmoid(d.squeeze(1)) > 0.8).long() * (a_prob > 0.8).long()

        return a_ind * z - (1 - z)

    def learn_routine(self, s, a, s_forget, a_forget):

        self.policy.train()
        self.discriminator.train()

        theta_snap_net = list(map(lambda x: x.clone().detach(), self.policy.parameters()))
        theta_snap_disc = list(map(lambda x: x.clone().detach(), self.discriminator.parameters()))

        real = torch.cuda.FloatTensor(len(s), 1).fill_(1)
        fake = torch.cuda.FloatTensor(len(s_forget), 1).fill_(0)

        # initialize augmented lagrange parameter
        mu_disc = .1
        mu_net = .1
        lagrange_net = 1.
        lagrange_disc = 1.

        acc_policy = 0
        acc_discriminator = 0

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

        for k in range(self.max_n_iter):

            # Policy part
            if acc_policy < self.min_acc or k < self.min_n_iter:

                theta_current_net = list(self.policy.parameters())

                theta_dist_net = 0.5 * torch.stack([((1 - mask) * (theta_i - theta_j) ** 2).sum() for theta_i, theta_j, mask
                                                in zip(theta_snap_net, theta_current_net, masks_policy)]).sum()

                a_est = self.policy(s)
                objective = self.loss_classification(a_est, a)

                # PA
                loss_classification = theta_dist_net + mu_net / 2 * objective ** 2 + lagrange_net * objective
                # ln = lagrange_net ** 0.5
                # loss_classification = theta_dist_net / ln + mu_net / (2 * ln) * objective ** 2 + objective * ln

                # loss_classification = theta_dist_net + objective

                self.optimizer_net.zero_grad()
                loss_classification.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40, norm_type=2)
                self.optimizer_net.step()

                lagrange_net += mu_net * objective.detach()

                acc_policy = (a == a_est.argmax(dim=1)).float().mean()

            if acc_discriminator < self.min_acc or k < self.min_n_iter:

                # MEM part

                theta_current_disc = list(self.discriminator.parameters())
                theta_dist_disc = 0.5 * torch.stack([((theta_i - theta_j) ** 2).sum() for theta_i, theta_j in zip(theta_snap_disc, theta_current_disc)]).sum()

                d_real, _, _, kl = self.discriminator(s, a)
                real_objective = self.loss_discrimination(d_real, real)
                prior_objective = kl.mean()

                d_fake, _, _, kl_forget = self.discriminator(s_forget, a_forget)
                fake_objective = self.loss_discrimination(d_fake, fake)
                prior_fake_objective = kl_forget.mean()

                objective = real_objective + fake_objective

                f_min = theta_dist_disc + prior_objective + prior_fake_objective
                # PA
                loss_discrimination = f_min + mu_disc / 2 * objective ** 2 + lagrange_disc * objective
                # lm = lagrange_disc ** 0.5
                # loss_discrimination = theta_dist_disc / lm + mu_disc / (2 * lm) * objective ** 2 + objective * lm
                # loss_discrimination = theta_dist_disc + objective

                self.optimizer_disc.zero_grad()
                loss_discrimination.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 40, norm_type=2)
                self.optimizer_disc.step()

                lagrange_disc += mu_disc * objective.detach()

                d = torch.cat([d_real, 1 - d_fake])
                acc_discriminator = (d > 0).float().mean()

        self.statistics['lagrange_disc'] = float(lagrange_disc)
        self.statistics['lagrange_net'] = float(lagrange_net)
        self.statistics['loss_discrimination'] = float(loss_discrimination)
        self.statistics['loss_classification'] = float(loss_classification)
        self.statistics['theta_dist_net'] = float(theta_dist_net)
        self.statistics['theta_dist_disc'] = float(theta_dist_disc)
        self.statistics['acc_policy'] = acc_policy
        self.statistics['acc_disc'] = acc_discriminator
        self.statistics['k_iter'] = k


