
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
        self.policy_stm = PolicyNet()
        self.memory_stm = PredictNet()

        if torch.cuda.device_count() > 1:
            self.policy_stm = nn.DataParallel(self.policy_stm)
            self.memory_stm = nn.DataParallel(self.memory_stm)

        self.policy_stm.to(self.device)
        self.policy_stm.train()

        self.memory_stm.to(self.device)
        self.memory_stm.train()

        self.optimizer_net = torch.optim.Adam(self.policy_stm.parameters(), lr=0.001, weight_decay=1e-5)
        self.optimizer_mem = torch.optim.Adam(self.memory_stm.parameters(), lr=0.001, weight_decay=1e-5)

        self.loss_classification = nn.CrossEntropyLoss()

        self.loss_classification_forget = nn.KLDivLoss(reduction='none')
        self.loss_reconstruction = nn.BCEWithLogitsLoss(reduction='none')

        self.n_iter = 20
        self.stm_batch = args.stm_batch

        self.learn_queue = []
        self.forget_queue = []
        self.action_queue = []

    def fifo_add(self, s, a=None, target='learn'):

        if target == 'learn':
            self.learn_queue.append(s)
        elif target == 'forget':
            self.forget_queue.append(s)
        else:
            raise NotImplementedError

        if a is not None:
            self.action_queue.append(a)

        if len(self.learn_queue) >= self.stm_batch:

            self.learn_routine(torch.stack(self.learn_queue),
                               torch.stack(self.action_queue),
                               torch.stack(self.forget_queue))

            self.learn_queue = []
            self.forget_queue = []
            self.action_queue = []

    def fetch_gate(self, s):

        self.policy_stm.train()
        self.memory_stm.train()

        a_hat = self.policy_stm(s)
        a_hat = a_hat.detach()
        a_ind = torch.argmax(a_hat, dim=1)

        mean, _, _ = self.memory_stm(s)
        mean = mean.detach()
        mean = mean.mean(dim=2)
        mean = mean.gather(1, a_ind.unsqueeze(1))
        a_hat = a_hat.gather(1, a_ind.unsqueeze(1))

        z = ((a_ind == torch.argmax(mean, dim=1)) * (mean > 0.5) * (a_hat > 0.75)).long()

        a_hat = a_ind * z - (1 - z)

        return a_hat

    def learn_gate(self, q_a, rr, v_0, v_n, tde_quantiles):

        r = np.sum(self.discount ** np.arange(self.n_steps) * rr)
        tde = r + self.discount ** self.n_steps * v_n - q_a

        return (np.sign(tde) * np.abs(tde / v_0)) > tde_quantiles

    def learn_routine(self, s, a, s_forget):

        self.policy_stm.train()
        self.memory_stm.train()

        theta_snap_net = list(map(lambda x: x.clone().detach(), self.policy_stm.parameters()))
        theta_snap_mem = list(map(lambda x: x.clone().detach(), self.memory_stm.parameters()))

        # initialize augmented lagrange parameter
        mu = 1.
        lagrange_net = 1.
        lagrange_mem = 1.

        for _ in range(self.n_iter):

            # Policy part

            theta_current_net = list(self.policy_stm.parameters())
            theta_dist_net = 0.5 * torch.stack([((theta_i - theta_j) ** 2).sum() for theta_i, theta_j in zip(theta_snap_net, theta_current_net)]).sum()

            a_est = self.policy_stm(s)
            objective = self.loss_classification(a_est, a)

            # PA
            loss_classification = theta_dist_net + mu / 2 * objective ** 2 + lagrange_net * objective

            self.optimizer_net.zero_grad()
            loss_classification.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_stm.parameters(), 1000, norm_type=2)
            self.optimizer_net.step()

            lagrange_net += mu * objective.detach()

            # MEM part

            theta_current_mem = list(self.memory_stm.parameters())
            theta_dist_mem = 0.5 * torch.stack([((theta_i - theta_j) ** 2).sum() for theta_i, theta_j in zip(theta_snap_mem, theta_current_mem)]).sum()

            s_reconstruct, _, _, kl = self.memory_stm(s, a)
            objective = (self.loss_reconstruction(s_reconstruct, s).mean(dim=0).sum() + kl.mean())

            _, _, kl_forget = self.memory_stm(s_forget)
            objective += kl_forget.mean()

            # PA
            loss_reconstruction = theta_dist_mem + mu / 2 * objective ** 2 + lagrange_mem * objective

            self.optimizer_mem.zero_grad()
            loss_reconstruction.backward()
            torch.nn.utils.clip_grad_norm_(self.memory_stm.parameters(), 1000, norm_type=2)
            self.optimizer_mem.step()

            lagrange_mem += mu * objective.detach()

