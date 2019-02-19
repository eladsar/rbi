# R2D2 implementation

import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn

from config import consts, args
import psutil
import socket

from model import DuelRNN

from memory_rnn import ObservationsRNNMemory, ObservationsRNNBatchSampler, collate
from agent import Agent
from environment import Env
from preprocess import release_file, lock_file, get_mc_value, get_td_value, h_torch, hinv_torch, get_expected_value, get_tde
import cv2
import os
import time
import shutil

imcompress = cv2.IMWRITE_PNG_COMPRESSION
compress_level = 2
mem_threshold = consts.mem_threshold


class R2D2Agent(Agent):

    def __init__(self, root_dir, player=False, choose=False, checkpoint=None):

        print("Learning with RBIRNNAgent")
        super(R2D2Agent, self).__init__(root_dir, checkpoint)

        self.value_net = DuelRNN()
        if torch.cuda.device_count() > 1:
            self.value_net = nn.DataParallel(self.value_net)
        self.value_net.to(self.device)

        self.pi_rand = np.ones(self.action_space) / self.action_space
        self.pi_rand_seq = torch.ones(self.batch, self.seq_length, self.action_space,  dtype=torch.float).to(self.device) / self.action_space
        self.pi_rand_bi = torch.ones(self.batch, self.burn_in, self.action_space, dtype=torch.float).to(self.device) / self.action_space

        self.a_zeros = torch.zeros(1, 1).long().to(self.device)
        self.a_zeros_bi = torch.zeros(self.batch, self.burn_in, 1, dtype=torch.long).to(self.device)

        self.q_loss = nn.SmoothL1Loss(reduction='none')

        if player:

            # play variables
            self.env = Env()
            self.a_zeros = torch.zeros(1, 1, 1).long().to(self.device)
            self.trajectory = []
            self.images = []
            self.choices = np.arange(self.action_space, dtype=np.int)
            self.n_replay_saved = 1
            self.frame = 0
            self.states = 0

        else:

            # datasets
            self.train_dataset = ObservationsRNNMemory(root_dir)
            self.train_sampler = ObservationsRNNBatchSampler(root_dir)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler, collate_fn=collate,
                                                            num_workers=args.cpu_workers, pin_memory=True, drop_last=False)

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.001, eps=1e-3, weight_decay=0)
        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        if torch.cuda.device_count() > 1:
            state = {'value_net': self.value_net.module.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'aux': aux}
        else:
            state = {'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        if torch.cuda.device_count() > 1:
            self.value_net.module.load_state_dict(state['value_net'])
            self.optimizer_value.load_state_dict(state['optimizer_value'])
        else:
            self.value_net.load_state_dict(state['value_net'])
            self.optimizer_value.load_state_dict(state['optimizer_value'])

        self.n_offset = state['aux']['n']
        return state['aux']

    def learn(self, n_interval, n_tot):

        target_net = DuelRNN()
        if torch.cuda.device_count() > 1:
            target_net = nn.DataParallel(target_net)
        target_net.to(self.device)

        target_net.load_state_dict(self.value_net.state_dict())

        self.value_net.train()
        target_net.eval()

        results = {'n': [], 'loss_value': [], 'loss_beta': [], 'act_diff': [], 'a_agent': [],
                   'a_player': [], 'loss_std': [], 'mc_val': [], "Hbeta": [], "Hpi": [], "adv_a": [], "q_a": [], 'image': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = sample['s'].to(self.device, non_blocking=True)
            a = sample['a'].to(self.device, non_blocking=True).unsqueeze_(2)
            # burn in
            h_q = sample['h_q'].to(self.device, non_blocking=True)
            s_bi = sample['s_bi'].to(self.device, non_blocking=True)
            r = sample['r'].to(self.device, non_blocking=True)
            t = sample['t'].to(self.device, non_blocking=True)
            R = sample['rho_q'].to(self.device, non_blocking=True)
            tde = sample['tde'].to(self.device, non_blocking=True)

            _, _, h_q = self.value_net(s_bi, self.a_zeros_bi, self.pi_rand_bi, h_q)

            q, q_a, _ = self.value_net(s, a, self.pi_rand_seq, h_q)
            a_tag = torch.argmax(q, dim=2).detach().unsqueeze(2)

            _, q_target, _ = target_net(s, a_tag, self.pi_rand_seq, h_q)
            q_target = q_target.detach()

            r = h_torch(r + self.discount ** self.n_steps * (1 - t[:, self.n_steps:]) * hinv_torch(q_target[:, self.n_steps:]))

            is_value = tde ** (-self.priority_beta)
            is_value = is_value / is_value.max()
            is_value = is_value.unsqueeze(1).repeat(1, self.seq_length - self.n_steps)

            loss_value = (self.q_loss(q_a[:, :-self.n_steps], r) * is_value * (1 - t[:, :-self.n_steps])).mean()

            # Learning part

            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

            # collect actions statistics

            if not (n + 1 + self.n_offset) % 10:

                if not (n + 1 + self.n_offset) % 50:

                    a_index_np = a[:, :-self.n_steps, 0].contiguous().view(-1).data.cpu().numpy()

                    q_a = q_a[:, :-self.n_steps].contiguous().view(-1).data.cpu().numpy()
                    r = r.view(-1).data.cpu().numpy()

                    _, beta_index = q[:, :-self.n_steps, :].contiguous().view(-1, self.action_space).data.cpu().max(1)
                    beta_index = beta_index.numpy()
                    act_diff = (a_index_np != beta_index).astype(np.int)

                    R = R.view(-1).data.cpu().numpy()

                    # add results

                    results['act_diff'].append(act_diff)
                    results['a_agent'].append(beta_index)
                    results['adv_a'].append(r)
                    results['q_a'].append(q_a)
                    results['a_player'].append(a_index_np)
                    results['Hbeta'].append(0)
                    results['Hpi'].append(0)
                    results['mc_val'].append(R)

                    # add results
                    results['loss_beta'].append(((R - r) ** 2).mean())
                    results['loss_value'].append(loss_value.data.cpu().numpy())
                    results['loss_std'].append(0)
                    results['n'].append(n)

                if not (n + 1 + self.n_offset) % self.update_target_interval:
                    # save agent state
                    target_net.load_state_dict(self.value_net.state_dict())

                if not (n + 1 + self.n_offset) % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': self.n_offset + n + 1})

                if not (n + 1 + self.n_offset) % n_interval:
                    results['act_diff'] = np.concatenate(results['act_diff'])
                    results['a_agent'] = np.concatenate(results['a_agent'])
                    results['adv_a'] = np.concatenate(results['adv_a'])
                    results['q_a'] = np.concatenate(results['q_a'])
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['mc_val'] = np.concatenate(results['mc_val'])
                    results['image'] = s[0, 0, :-1, :, :].data.cpu()

                    yield results
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if (n + self.n_offset) >= n_tot:
                        break

            del loss_value

    def play(self, n_tot, save=True, load=True, fix=False):

        pi_rand_t = torch.ones(1, 1, self.action_space, dtype=torch.float).to(self.device) / self.action_space

        for i in range(n_tot):

            self.env.reset()

            rewards = [[]]
            v_target = [[]]
            q_val = []
            lives = self.env.lives

            while not fix:
                try:
                    self.load_checkpoint(self.snapshot_path)
                    break
                except:
                    time.sleep(0.5)

            self.value_net.eval()

            # Initial states
            h_q = torch.zeros(1, self.hidden_state).to(self.device)

            while not self.env.t:

                if load and not (self.states % self.load_memory_interval):
                    try:
                        self.load_checkpoint(self.snapshot_path)
                    except:
                        pass

                    self.value_net.eval()

                s = self.env.s.to(self.device).unsqueeze(0)
                # take q as adv

                q, _, h_q = self.value_net(s, self.a_zeros, pi_rand_t, h_q)

                q = q.squeeze(0).squeeze(0).data.cpu().numpy()

                if self.n_offset >= self.random_initialization:

                    pi = np.zeros(self.action_space)
                    pi[np.argmax(q)] = 1
                    pi_mix = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi
                else:
                    pi_mix = self.pi_rand

                pi_mix = pi_mix.clip(0, 1)
                pi_mix = pi_mix / pi_mix.sum()
                a = np.random.choice(self.choices, p=pi_mix)
                self.env.step(a)

                if self.env.k >= self.history_length:

                    if lives > self.env.lives:
                        rewards.append([])
                        v_target.append([])
                    lives = self.env.lives

                    rewards[-1].append(self.env.r)
                    v_target[-1].append(0)
                    q_val.append(q[a])

                self.frame += 1

            mc_val = get_mc_value(rewards, None, self.discount, None)
            q_val = np.array(q_val)

            print("sts | st: %d\t| sc: %d\t| f: %d\t| e: %7g\t| typ: %2d | trg: %d | nst: %s\t| n %d\t| avg_r: %g\t| avg_f: %g" %
                (self.frame, self.env.score, self.env.k, 0, 0, 0, str(np.nan),
                self.n_offset, self.behavioral_avg_score, self.behavioral_avg_frame))

            yield {'score': self.env.score,
                   'frames': self.env.k, "n": self.n_offset, "mc": mc_val, "q": q_val}

            if self.n_offset >= self.n_tot and not fix:
                break

        raise StopIteration

    def multiplay(self):

        n_players = self.n_players
        pi_rand_t = torch.ones(n_players, 1, self.action_space, dtype=torch.float).to(self.device) / self.action_space

        player_i = np.arange(self.actor_index, self.actor_index + self.n_actors * n_players, self.n_actors) / (self.n_actors * n_players - 1)
        explore_threshold = player_i

        mp_explore = 0.4 ** (1 + 7 * (1 - player_i))

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0

        a_zeros_mp = self.a_zeros.repeat(n_players, 1, 1)
        mp_pi_rand = np.repeat(np.expand_dims(self.pi_rand, axis=0), n_players, axis=0)

        rewards = [[[]] for _ in range(n_players)]
        v_target = [[[]] for _ in range(n_players)]
        q_expected = [[] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        image_dir = ['' for _ in range(n_players)]
        trajectory = [[] for _ in range(n_players)]
        screen_dir = [os.path.join(self.explore_dir, "screen")] * n_players
        fr_s = [self.frame + self.history_length - 1 for _ in range(n_players)]

        trajectory_dir = [os.path.join(self.explore_dir, "trajectory")] * n_players
        readlock = [os.path.join(self.list_dir, "readlock_explore.npy")] * n_players

        # set initial episodes number
        # lock read
        fwrite = lock_file(self.episodelock)
        current_num = np.load(fwrite).item()
        episode_num = current_num + np.arange(n_players)
        fwrite.seek(0)
        np.save(fwrite, current_num + n_players)
        # unlock file
        release_file(fwrite)

        # Initial states
        h_q = torch.zeros(n_players, self.hidden_state).to(self.device)

        for i in range(n_players):
            mp_env[i].reset()
            image_dir[i] = os.path.join(screen_dir[i], str(episode_num[i]))
            os.mkdir(image_dir[i])

        lives = [mp_env[i].lives for i in range(n_players)]

        while True:

            if not (self.frame % self.load_memory_interval):
                try:
                    self.load_checkpoint(self.snapshot_path)
                except:
                    pass

                self.value_net.eval()

            # save previous hidden state to np object
            h_q_np = h_q.data.cpu().numpy()

            s = torch.cat([env.s for env in mp_env]).to(self.device).unsqueeze(1)

            # take q as adv
            q, _, h_q = self.value_net(s, a_zeros_mp, pi_rand_t, h_q)

            q = q.squeeze(1).data.cpu().numpy()

            mp_trigger = np.logical_and(
                np.array([env.score for env in mp_env]) >= self.behavioral_avg_score * explore_threshold,
                explore_threshold >= 0)
            exploration = np.repeat(np.expand_dims(mp_explore * mp_trigger, axis=1), self.action_space, axis=1)

            if self.n_offset >= self.random_initialization:

                pi = np.zeros((n_players, self.action_space))
                pi[range(n_players), np.argmax(q, axis=1)] = 1

            else:
                pi = mp_pi_rand

            pi_mix = pi * (1 - exploration) + exploration * mp_pi_rand

            pi_mix = pi_mix.clip(0, 1)
            pi_mix = pi_mix / np.repeat(pi_mix.sum(axis=1, keepdims=True), self.action_space, axis=1)

            pi = pi.astype(np.float32)

            for i in range(n_players):

                a = np.random.choice(self.choices, p=pi_mix[i])

                env = mp_env[i]
                cv2.imwrite(os.path.join(image_dir[i], "%s.png" % str(self.frame)), mp_env[i].image, [imcompress, compress_level])

                h_beta_save = np.zeros_like(h_q_np[i]) if not self.frame % self.seq_overlap else None
                h_q_save = h_q_np[i] if not self.frame % self.seq_overlap else None

                episode[i].append(np.array((self.frame, a, pi[i],
                                            h_beta_save, h_q_save,
                                            episode_num[i], 0., fr_s[i], 0,
                                            0., 1., 1., 0, 1., 0), dtype=self.rec_type))
                env.step(a)

                if lives[i] > env.lives:
                    rewards[i].append([])
                    v_target[i].append([])
                lives[i] = env.lives

                rewards[i][-1].append(env.r)
                v_target[i][-1].append(q[i].max())
                q_expected[i].append(q[i][a])

                if env.t:

                    # cancel termination reward
                    rewards[i][-1][-1] -= self.termination_reward * int(env.k * self.skip >= self.max_length or env.score >= self.max_score)
                    td_val = get_expected_value(rewards[i], v_target[i], self.discount, self.n_steps)

                    episode_df = np.stack(episode[i][self.history_length - 1:self.max_length])

                    tde = get_tde(rewards[i], v_target[i], self.discount, self.n_steps, q_expected[i])
                    episode_df['tde'] = tde[self.history_length - 1:self.max_length]

                    mc_val = get_mc_value(rewards[i], v_target[i], self.discount, self.n_steps)

                    episode_df['r'] = td_val[self.history_length - 1:self.max_length]

                    # hack to save the true target (MC value)
                    episode_df['rho_q'] = mc_val[self.history_length - 1:self.max_length]

                    episode_df['fr_e'] = episode_df[-1]['fr'] + 1
                    trajectory[i].append(episode_df)

                    # reset hidden states
                    h_q[i, :].zero_()

                    print("rbi | st: %d\t| sc: %d\t| f: %d\t| e: %7g\t| typ: %2d | trg: %d | t: %d\t| n %d\t| avg_r: %g\t| avg_f: %g" %
                          (self.frame, env.score, env.k, mp_explore[i], np.sign(explore_threshold[i]), 1, time.time() - self.start_time, self.n_offset, self.behavioral_avg_score, self.behavioral_avg_frame))

                    env.reset()
                    episode[i] = []
                    q_expected[i] = []
                    rewards[i] = [[]]
                    v_target[i] = [[]]
                    lives[i] = env.lives
                    fr_s[i] = (self.frame + 1) + (self.history_length - 1)

                    # get new episode number

                    # lock read
                    fwrite = lock_file(self.episodelock)
                    episode_num[i] = np.load(fwrite).item()
                    fwrite.seek(0)
                    np.save(fwrite, episode_num[i] + 1)
                    # unlock file
                    release_file(fwrite)

                    image_dir[i] = os.path.join(screen_dir[i], str(episode_num[i]))
                    os.mkdir(image_dir[i])

                    if sum([len(j) for j in trajectory[i]]) >= self.player_replay_size:

                        # write if enough space is available
                        if psutil.virtual_memory().available >= mem_threshold:

                            # lock read
                            fwrite = lock_file(self.writelock)
                            traj_num = np.load(fwrite).item()
                            fwrite.seek(0)
                            np.save(fwrite, traj_num + 1)
                            # unlock file
                            release_file(fwrite)

                            traj_to_save = np.concatenate(trajectory[i])
                            traj_to_save['traj'] = traj_num

                            traj_file = os.path.join(trajectory_dir[i], "%d.npy" % traj_num)
                            np.save(traj_file, traj_to_save)

                            fread = lock_file(readlock[i])
                            traj_list = np.load(fread)
                            fread.seek(0)
                            np.save(fread, np.append(traj_list, traj_num))

                            release_file(fread)

                        trajectory[i] = []

            # write trajectory to dir

            self.frame += 1
            if not self.frame % self.player_replay_size:
                yield True
                if self.n_offset >= self.n_tot:
                    break

    def demonstrate(self, n_tot):

        self.value_net.eval()

        for i in range(n_tot):

            if "gpu" in socket.gethostname():
                log_dir = os.path.join("/home/dsi/elad/data/rbi/runs", "%s_%d" % (consts.exptime, i))
            else:
                log_dir = os.path.join("/tmp", "%s_%d" % (consts.exptime, i))

            os.mkdir(log_dir)

            self.env.reset()

            # here there is a problem when there is a varying/increasing life counter as in mspacman

            choices = np.arange(self.action_space, dtype=np.int)

            while not self.env.t:

                s = self.env.s.to(self.device)
                aux = self.env.aux.to(self.device)

                beta = self.beta_net(s, aux)

                beta_softmax = F.softmax(beta, dim=2)

                v, adv, _, q, _ = self.value_net(s, self.a_zeros, beta_softmax, aux)
                v = v.squeeze(0)
                adv = adv.squeeze(0)
                q = q.squeeze(0).data.cpu().numpy()

                beta = beta.squeeze(0)
                beta = F.softmax(beta, dim=2)
                beta = beta.data.cpu().numpy()


                if False:

                    pi = beta.copy()
                    adv2 = adv.copy()

                    rank = np.argsort(adv2)
                    adv_rank = np.argsort(rank).astype(np.float)

                    pi = self.cmin * pi

                    Delta = 1 - self.cmin
                    while Delta > 0:
                        a = np.argmax(adv2)
                        Delta_a = np.min((Delta, (self.cmax - self.cmin) * beta[a]))
                        Delta -= Delta_a
                        pi[a] += Delta_a
                        adv2[a] = -1e11

                    # pi_adv = 2 ** adv_rank * np.logical_or(adv2 >= 0, adv_rank == (self.action_space - 1))
                    pi_adv = 1. * np.logical_or(adv >= 0, adv_rank == (self.action_space - 1))
                    pi_adv = pi_adv / (np.sum(pi_adv))

                    pi = (1 - self.mix) * pi + self.mix * pi_adv
                    pi_mix = self.eps_pre * self.pi_rand + (1 - self.eps_pre) * pi

                else:
                    pi = beta

                pi = pi.clip(0, 1)
                pi = pi / pi.sum()

                a = np.random.choice(choices, p=pi)
                self.env.step(a)

                # time.sleep(0.1)

                img = s.squeeze(0).data[:3].cpu().numpy()
                img = np.rollaxis(img, 0, 3)[:, :, :3]
                img = (img * 256).astype(np.uint8)

                cv2.imwrite(os.path.join(log_dir, "%d_%d_%d.png" % (self.env.k, a, self.env.score)), img)

                yield {'score': self.env.score,
                       "beta": pi,
                       "v": v.data.cpu().numpy(),
                       "q": q,
                       "aux": aux.squeeze(0).data.cpu().numpy(),
                       "adv": adv.data.cpu().numpy(),
                       "o": img,
                       'frames': self.env.k,
                       "actions": self.env.action_meanings,
                       "a": a
                       }

        raise StopIteration