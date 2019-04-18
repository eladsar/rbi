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
from model import BehavioralNet, DuelNet, PredictNet

from memory import ReplayBatchSampler, Memory, collate
from agent import Agent
from environment import Env
from preprocess import get_tde_value, get_mc_value, h_torch, hinv_torch, \
                       release_file, lock_file, get_td_value, state_to_img, kl_gaussian, h_gaussian

import scipy.optimize as opt

import cv2
import os
import time

imcompress = cv2.IMWRITE_PNG_COMPRESSION
compress_level = 2
mem_threshold = consts.mem_threshold


class RBIAgent(Agent):

    def __init__(self, root_dir, player=False, choose=False, checkpoint=None):

        print("Learning with RBIAgent")
        super(RBIAgent, self).__init__(root_dir, checkpoint, player)

        self.beta_net = self.net_to_gpu(BehavioralNet())
        self.value_net = self.net_to_gpu(DuelNet())
        self.target_net = self.net_to_gpu(DuelNet())
        self.predict_net = self.net_to_gpu(PredictNet())

        self.target_net.load_state_dict(self.value_net.state_dict())

        self.pi_rand = np.ones(self.action_space) / self.action_space
        self.pi_rand_batch = torch.FloatTensor(self.pi_rand).unsqueeze(0).repeat(self.batch, 1).to(self.device)

        self.a_zeros = torch.zeros(1, 1).long().to(self.device)
        self.a_zeros_batch = self.a_zeros.repeat(self.batch, 1)

        self.q_loss = nn.SmoothL1Loss(reduction='none')
        self.pred_loss = nn.BCELoss(reduction='none')

        if player:

            # play variables
            self.env = Env()
            self.a_zeros = torch.zeros(1, 1).long().to(self.device)
            self.trajectory = []
            self.images = []
            self.choices = np.arange(self.action_space, dtype=np.int)
            self.n_replay_saved = 1
            self.frame = 0
            self.states = 0

        else:

            self.train_dataset = Memory(root_dir)
            self.train_sampler = ReplayBatchSampler(root_dir)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            collate_fn=collate, num_workers=args.cpu_workers,
                                                            pin_memory=True, drop_last=False)
        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.optimizer_beta = torch.optim.Adam(self.beta_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.optimizer_predict = torch.optim.Adam(self.predict_net.parameters(), lr=0.00025 / 4, eps=1.5e-4, weight_decay=0)

        self.n_offset = 0

    def save_checkpoint(self, path, aux=None, update_predict=True):

        state = {'beta_net': Agent.state_dict(self.beta_net),
                 'value_net': Agent.state_dict(self.value_net),
                 'target_net': Agent.state_dict(self.target_net),
                 'optimizer_value': self.optimizer_value.state_dict(),
                 'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        if update_predict:
            state['predict_net'] = Agent.state_dict(self.predict_net)
            state['optimizer_predict'] = self.optimizer_predict.state_dict()

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        Agent.load_state_dict(self.beta_net, state['beta_net'])
        Agent.load_state_dict(self.value_net, state['value_net'])
        Agent.load_state_dict(self.target_net, state['target_net'])

        if 'predict_net' in state:
            Agent.load_state_dict(self.predict_net, state['predict_net'])
            self.optimizer_predict.load_state_dict(state['optimizer_predict'])

        self.optimizer_beta.load_state_dict(state['optimizer_beta'])
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        try:
            self.behavioral_avg_score = state['aux']['score']
        except:
            pass

        return state['aux']

    def learn(self, n_interval, n_tot):

        self.beta_net.train()
        self.value_net.train()
        self.predict_net.train()
        self.target_net.eval()

        results = {'n': [], 'loss_value': [], 'loss_beta': [], 'act_diff': [], 'a_agent': [],
                   'a_player': [], 'loss_std': [],
                   'mc_val': [], "Hbeta": [], "Hpi": [], "adv_a": [], "q_a": [], 'image': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            n = self.n_offset + n + 1

            s = sample['s'].to(self.device)
            s_tag = sample['s_tag'].to(self.device)
            a = sample['a'].to(self.device).unsqueeze_(1)
            r = sample['r'].to(self.device)
            t = sample['t'].to(self.device)
            pi = sample['pi'].to(self.device)
            pi_tag = sample['pi_tag'].to(self.device)
            tde = sample['tde'].to(self.device)

            # predict net

            # sd = torch.cat((s, s), dim=0)
            # sd_tag = torch.cat((s_tag, s_tag), dim=0)
            # ad = torch.cat((a+1, self.a_zeros_batch), dim=0)
            sd_pred, mean, var = self.predict_net(s, a)

            # sd_target = ((sd_tag - sd).abs() > 0).float()
            # sd_target = ((sd[:,1:,:,:] - sd[:,:-1,:,:]).abs() > 0).float()
            sd_target = ((s[:, 1:, :, :] - s[:, :-1, :, :]).abs() > 0).float()

            err1 = self.pred_loss(sd_pred, sd_target)

            err1 = err1.view(self.batch * 2, -1).sum(dim=1).mean()
            err2 = 0.5 * (mean ** 2 + var - torch.log(var) - 1).sum(dim=1).mean()

            loss_pred = (err1 + err2) / 10

            # loss_pred = ((sd_pred - sd_tag) ** 2).view(self.batch * 2, -1).sum(dim=1).mean() \
            #             + 0.5 * (mean ** 2 + var - torch.log(var)).sum(dim=1).mean()

            if not n % 100:
                print("\nelbo")
                print(err1)
                print("free energy")
                print(err2)
                print("mean stats")
                print(mean.mean())
                print(mean.max())
                print(mean.min())
                print("var stats")
                print(var.mean())
                print(var.max())
                print(var.min())
                print("sd_target stats")
                print(sd_target.max())
                print(sd_target.min())
                print("sd_pred stats")
                print(sd_pred.max())
                print(sd_pred.min())

            # image = sd_target
            image = sd_pred
            # print(image)
            # image = (sd_tag[:, 1:, :, :]-sd_tag[:, :-1, :, :])

            # Behavioral nets
            beta = self.beta_net(s)
            beta_log = F.log_softmax(beta, dim=1)
            beta = F.softmax(beta.detach(), dim=1)

            _, _, _, q, q_a = self.value_net(s, a, self.pi_rand_batch)
            _, _, _, q_tag, _ = self.target_net(s_tag, a, self.pi_rand_batch)

            q = q.detach()
            v_eval = (q * pi).sum(dim=1).abs()
            v_target = (q_tag * pi_tag).sum(dim=1).detach()

            r = h_torch(r + self.discount ** self.n_steps * (1 - t) * hinv_torch(v_target))

            is_value = tde ** (-self.priority_beta)
            is_policy = is_value

            is_value = is_value / is_value.max()

            v_diff = (q * (beta - pi)).abs().sum(dim=1)
            # is_policy = (torch.min(v_diff, v_diff/v_eval) / tde) ** self.priority_alpha
            is_policy = is_policy * torch.min(v_diff, v_diff/v_eval) ** self.priority_alpha

            is_policy = is_policy / is_policy.max()

            loss_value = (self.q_loss(q_a, r) * is_value).mean()
            loss_beta = ((-pi * beta_log).sum(dim=1) * is_policy).mean()

            # Learning part

            self.optimizer_beta.zero_grad()
            loss_beta.backward()
            self.optimizer_beta.step()

            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

            self.optimizer_predict.zero_grad()
            loss_pred.backward()
            self.optimizer_predict.step()

            # collect actions statistics
            if not n % 50:

                update_predict = False
                a_index_np = a[:, 0].data.cpu().numpy()

                # avoid zero pi
                pi = pi.clamp(min=1e-4, max=1)
                pi /= pi.sum(dim=1).unsqueeze(1).repeat(1, self.action_space)

                pi_log = pi.log()

                Hpi = -(pi * pi_log).sum(dim=1)
                Hbeta = -(beta * beta_log).sum(dim=1)

                adv_a = r.data.cpu().numpy()
                q_a = q_a.data.cpu().numpy()
                r = r.data.cpu().numpy()

                _, beta_index = beta.data.cpu().max(1)
                beta_index = beta_index.numpy()
                act_diff = (a_index_np != beta_index).astype(np.int)

                # add results

                results['act_diff'].append(act_diff)
                results['a_agent'].append(beta_index)
                results['adv_a'].append(adv_a)
                results['q_a'].append(q_a)
                results['a_player'].append(a_index_np)
                results['Hbeta'].append(Hbeta.data.mean().cpu().numpy())
                results['Hpi'].append(Hpi.data.mean().cpu().numpy())
                results['mc_val'].append(r)

                # add results
                results['loss_beta'].append(loss_beta.data.cpu().numpy())
                results['loss_value'].append(loss_value.data.cpu().numpy())
                results['loss_std'].append(loss_pred.data.cpu().numpy())
                results['n'].append(n)

                if not n % self.update_target_interval:
                    # save agent state
                    update_predict = True
                    self.target_net.load_state_dict(self.value_net.state_dict())

                if not n % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': n}, update_predict=update_predict)

                if not n % n_interval:
                    results['act_diff'] = np.concatenate(results['act_diff'])
                    results['a_agent'] = np.concatenate(results['a_agent'])
                    results['adv_a'] = np.concatenate(results['adv_a'])
                    results['q_a'] = np.concatenate(results['q_a'])
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['mc_val'] = np.concatenate(results['mc_val'])
                    results['image'] = image[0, :3, :, :].data.cpu()

                    yield results
                    self.beta_net.train()
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        break

    def play(self, n_tot, save=True, load=True, fix=False):

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

            self.beta_net.eval()
            self.value_net.eval()
            self.predict_net.train()

            while not self.env.t:

                if load and not (self.states % self.load_memory_interval):
                    try:
                        self.load_checkpoint(self.snapshot_path)
                    except:
                        pass

                    self.beta_net.eval()
                    self.value_net.eval()

                s = self.env.s.to(self.device)
                # get aux data

                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)

                # take q as adv

                v, adv, _, q, _ = self.value_net(s, self.a_zeros, beta)

                q = q.squeeze(0).data.cpu().numpy()
                v_expected = v.squeeze(0).data.cpu().numpy()

                beta = beta.squeeze(0).data.cpu().numpy()
                adv = adv.squeeze(0).data.cpu().numpy()

                if self.n_offset >= self.random_initialization:

                    if self.player == "reroutetv":

                        pi = beta.copy()
                        adv2 = adv.copy()

                        pi = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi

                        pi = self.cmin * pi

                        Delta = 1 - self.cmin
                        while Delta > 0:
                            a = np.argmax(adv2)
                            Delta_a = np.min((Delta, (self.cmax - self.cmin) * beta[a]))
                            Delta -= Delta_a
                            pi[a] += Delta_a
                            adv2[a] = -1e11

                        pi_greed = np.zeros(self.action_space)
                        pi_greed[np.argmax(adv)] = 1

                        pi_mix = (1 - self.mix) * pi + self.mix * pi_greed

                    elif self.player == "behavioral":
                        pi_mix = beta.copy()

                    else:
                        raise NotImplementedError

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
                    v_target[-1].append(v_expected)
                    q_val.append(q[a])

                self.frame += 1

            mc_val = get_mc_value(rewards, None, self.discount, None)
            q_val = np.array(q_val)

            print("sts | st: %d\t| [sc: %d\t| f: %d\t| e: %7g\t| typ: %2d | trg: %d | nst: %s\t| n %d\t| avg_r: %g\t| avg_f: %g\t| min: %g\t|" %
                (self.frame, self.env.score, self.env.k, 0, 0, 0, str(np.nan),
                self.n_offset, self.behavioral_avg_score, self.behavioral_avg_frame, self.env.min_score))

            yield {'score': self.env.score, 'min_score': self.env.min_score,
                   'frames': self.env.k, "n": self.n_offset, "mc": mc_val, "q": q_val}

            if self.n_offset >= self.n_tot and not fix:
                break

        raise StopIteration

    def multiplay(self):

        n_players = self.n_players

        # player_i = np.arange(self.actor_index, self.actor_index + self.n_actors * n_players, self.n_actors) / (self.n_actors * n_players - 1)
        # explore_i = 0.4 ** (1 + 7 * (1 - player_i))
        # explore_threshold = player_i
        # mp_explore = explore_i

        A_ub = np.eye(self.action_space)
        A_eq = np.ones((1, self.action_space))
        b_eq = np.ones((1,)) * (1 - self.cmin)

        # A_ub = np.eye(n_players * self.action_space)

        players = np.arange(self.n_actors) / (self.n_actors - 1)
        explorers = 0.4 ** (1 + 7 * (1 - players))

        roll_player = np.random.choice(self.n_actors, size=n_players)
        explore_threshold = players[roll_player]
        mp_explore = explorers[roll_player] * np.random.choice(2, size=n_players, p=[0.2, 0.8])

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0

        a_zeros_mp = self.a_zeros.repeat(n_players, 1)

        mp_pi_rand = np.repeat(np.expand_dims(self.pi_rand, axis=0), n_players, axis=0)

        range_players = np.arange(n_players)
        rewards = [[[]] for _ in range(n_players)]
        v_target = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        q_a = [[] for _ in range(n_players)]
        rewards_kl = [[] for _ in range(n_players)]
        image_dir = ['' for _ in range(n_players)]
        trajectory = [[] for _ in range(n_players)]
        screen_dir = [os.path.join(self.explore_dir, "screen")] * n_players

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

                self.beta_net.eval()
                self.value_net.eval()
                self.predict_net.train()

            s = torch.cat([env.s for env in mp_env]).to(self.device)

            beta = self.beta_net(s)
            beta = F.softmax(beta.detach(), dim=1)
            # take q as adv
            _, adv, _, q, _ = self.value_net(s, a_zeros_mp, beta)
            beta = beta.data.cpu().numpy()

            q = q.data.cpu().numpy()
            adv = adv.data.cpu().numpy()

            threshold = self.behavioral_min_score + (self.behavioral_avg_score - self.behavioral_min_score) * explore_threshold
            mp_trigger = np.logical_or(np.array([env.score for env in mp_env]) >= threshold, explore_threshold == 0)
            exploration = np.repeat(np.expand_dims(mp_explore * mp_trigger, axis=1), self.action_space, axis=1)

            beta = (1 - self.epsilon) * beta + self.epsilon / self.action_space
            pi = self.cmin * beta

            if self.n_offset >= self.random_initialization:

                for i in range(n_players):
                    b_ub = (self.cmax - self.cmin) * beta[i]
                    x = opt.linprog(-adv[i], A_ub, b_ub, A_eq, b_eq)
                    pi[i] += x.x


                pi_greed = np.zeros((n_players, self.action_space))
                pi_greed[range(n_players), np.argmax(adv, axis=1)] = 1
                pi = (1 - self.mix) * pi + self.mix * pi_greed

            else:
                pi = mp_pi_rand

            pi_mix = pi * (1 - exploration) + exploration * mp_pi_rand
            pi_mix = pi_mix.clip(0, 1)
            pi_mix = pi_mix / np.repeat(pi_mix.sum(axis=1, keepdims=True), self.action_space, axis=1)

            pi = pi.astype(np.float32)

            v_expected = (q * pi).sum(axis=1)

            a = [np.random.choice(self.choices, p=pi_mix[i]) for i in range(n_players)]
            # a_torch = torch.LongTensor(a).unsqueeze(1).to(self.device)

            # sd = torch.cat((s, s), dim=0)
            # ad = torch.cat((a_torch + 1, a_zeros_mp), dim=0)

            # add aux reward
            sd_pred, mean, var = self.predict_net(s, a)
            mean = mean.detach().cpu().numpy()
            var = var.detach().cpu().numpy()

            # r_kl1 = (sd_pred[:n_players] - sd_pred[n_players:]).abs().detach().view(n_players, -1).sum(dim=1).cpu().numpy()
            # r_kl2 = kl_gaussian(mean[:n_players], var[:n_players], mean[n_players:], var[n_players:])

            s_target = ((s[:,1:,:,:] - s[:,:-1,:,:]).abs() > 0).float()

            r_kl = self.pred_loss(sd_pred, s_target).detach().view(n_players, -1).sum(dim=1).cpu().numpy()
            r_kl -= 0.5 * (mean ** 2 + var - np.log(var) - 1).sum(axis=1)

            # r_kl = self.pred_loss(sd_pred[n_players:], s_target).detach().view(n_players, -1).mean(dim=1).cpu().numpy() * 100
            if self.n_offset >= self.random_initialization * 2:

                # r_kl = h_gaussian(mean[n_players:], var[n_players:])

                r_kl = r_kl * (1 - self.discount) ** 2

            else:
                r_kl = r_kl * 0

            for i in range(n_players):

                env = mp_env[i]
                cv2.imwrite(os.path.join(image_dir[i], "%s.png" % str(self.frame)), mp_env[i].image, [imcompress, compress_level])
                episode[i].append(np.array((self.frame, a[i], pi[i],
                                            None, None,
                                            episode_num[i], 0., 0, 0,
                                            0., 1., 1., 0, 1., 0), dtype=self.rec_type))

                env.step(a[i])

                if lives[i] > env.lives:
                    rewards[i].append([])
                    v_target[i].append([])
                lives[i] = env.lives

                rewards[i][-1].append(env.r + r_kl[i])
                v_target[i][-1].append(v_expected[i])
                q_a[i].append(q[i][a[i]])
                rewards_kl[i].append(r_kl[i])

                if env.t:

                    # cancel termination reward
                    rewards[i][-1][-1] -= self.termination_reward * int(env.k * self.skip >= self.max_length or env.score >= self.max_score)

                    td_val, t_val = get_tde_value(rewards[i], self.discount, self.n_steps)
                    tde = np.abs(np.array(q_a[i]) - get_td_value(rewards[i], v_target[i], self.discount, self.n_steps))
                    v_scale = np.concatenate(v_target[i])

                    # tde = ((tde + 0.01) / (np.abs(v_scale) + 0.01)) ** self.priority_alpha
                    tde = np.minimum(tde, tde / np.abs(v_scale)) ** self.priority_alpha

                    episode_df = np.stack(episode[i][self.history_length - 1:self.max_length])
                    episode_df['r'] = td_val[self.history_length - 1:self.max_length]
                    episode_df['t'] = t_val[self.history_length - 1:self.max_length]
                    episode_df['tde'] = tde[self.history_length - 1:self.max_length]

                    trajectory[i].append(episode_df)

                    print("rbi | st: %d\t| sc: %d\t| f: %d\t| e: %7g\t| typ: %2d | trg: %d | t: %d\t| n %d\t| avg_r: %g\t| avg_f: %g\t| min: %g\t| r_kl: %g\t|" %
                          (self.frame, env.score, env.k, mp_explore[i],  np.sign(explore_threshold[i]),
                           mp_trigger[i], time.time() - self.start_time, self.n_offset, self.behavioral_avg_score,
                           self.behavioral_avg_frame, env.min_score, np.mean(rewards_kl[i])))

                    env.reset()
                    episode[i] = []
                    q_a[i] = []
                    rewards_kl[i] = []
                    rewards[i] = [[]]
                    v_target[i] = [[]]
                    lives[i] = env.lives

                    roll_player = np.random.choice(self.n_actors)
                    explore_threshold[i] = players[roll_player] * np.random.choice(2, p=[0.2, 0.8])
                    mp_explore[i] = explorers[roll_player]

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

        self.beta_net.eval()
        self.value_net.eval()

        for i in range(n_tot):

            # if "gpu" in socket.gethostname():
            #     log_dir = os.path.join("/home/dsi/elad/data/rbi/runs", "%s_%d" % (consts.exptime, i))
            # else:
            #     log_dir = os.path.join("/tmp", "%s_%d" % (consts.exptime, i))
            #
            # os.mkdir(log_dir)

            self.env.reset()

            # here there is a problem when there is a varying/increasing life counter as in mspacman

            choices = np.arange(self.action_space, dtype=np.int)

            while not self.env.t:

                s = self.env.s.to(self.device)
                beta = self.beta_net(s)
                beta = F.softmax(beta.detach(), dim=1)

                _, _, _, q, _ = self.value_net(s, self.a_zeros, beta)

                q = q.squeeze(0).data.cpu().numpy()
                beta = beta.squeeze(0).data.cpu().numpy()

                pi = beta.copy()
                adv = q.copy()

                pi = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi
                pi = self.cmin * pi

                Delta = 1 - self.cmin
                while Delta > 0:
                    a = np.argmax(adv)
                    Delta_a = np.min((Delta, (self.cmax - self.cmin) * beta[a]))
                    Delta -= Delta_a
                    pi[a] += Delta_a
                    adv[a] = -1e11

                pi_greed = np.zeros(self.action_space)
                pi_greed[np.argmax(q)] = 1
                pi = (1 - self.mix) * pi + self.mix * pi_greed

                pi = pi.clip(0, 1)
                pi = pi / pi.sum()

                a = np.random.choice(choices, p=pi)
                self.env.step(a)

                # time.sleep(0.1)
                img = state_to_img(s)

                # cv2.imwrite(os.path.join(log_dir, "%d_%d_%d.png" % (self.env.k, a, self.env.score)), img)

                v = (pi * q).sum()
                adv = q - v

                yield {'score': self.env.score, "beta": pi, "v": v, "q": q, "adv": adv, "s": img, 'frames': self.env.k,
                       "actions": self.env.action_meanings, "a": a
                       }

        raise StopIteration