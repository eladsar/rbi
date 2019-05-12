import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import psutil

from config import consts, args
from model import DuelNet
from model_stm import PolicyNet, PredictNet

from memory import ReplayBatchSampler, Memory, collate
from agent import Agent
from environment import Env
from preprocess import get_tde_value, lock_file, release_file, h_torch, hinv_torch, get_mc_value, get_td_value, state_to_img
import cv2
import os
import time
import math

imcompress = cv2.IMWRITE_PNG_COMPRESSION
compress_level = 2
mem_threshold = consts.mem_threshold


class ApeAgent(Agent):

    def __init__(self, root_dir, player=False, choose=False, checkpoint=None):

        print("Learning with Ape Agent")
        super(ApeAgent, self).__init__(root_dir, checkpoint, player)

        self.value_net = DuelNet()
        self.target_net = DuelNet()

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

        self.weight = torch.FloatTensor(exp.class_weight) if args.balance else torch.ones(exp.labels_num)
        self.weight = self.weight.to(self.device)
        self.loss_classification = nn.CrossEntropyLoss(weight=self.weight)

        self.loss_classification_forget = nn.KLDivLoss(reduction='none')
        self.loss_reconstruction = nn.BCEWithLogitsLoss(reduction='none')

        self.punctuation = 1 - math.e ** (-5 / args.epoch)

        self.net_state_0 = list(map(lambda x: x.clone().detach(), self.policy_stm.parameters()))
        self.net_masks = list(map(lambda x: torch.cuda.FloatTensor(x.shape), self.policy_stm.parameters()))
        self.net_punct_list = list(map(lambda x: int(0), self.policy_stm.named_parameters()))

        self.mem_state_0 = list(map(lambda x: x.clone().detach(), self.memory_stm.parameters()))
        self.mem_masks = list(map(lambda x: torch.cuda.FloatTensor(x.shape), self.memory_stm.parameters()))
        self.mem_punct_list = list(map(lambda x: int('mu' in x[0] or 'rho' in x[0]), self.memory_stm.named_parameters()))

        self.n_iter_net = 20
        self.n_iter_mem = 20

        # back to dqn stuff

        if torch.cuda.device_count() > 1:
            self.value_net = nn.DataParallel(self.value_net)
            self.target_net = nn.DataParallel(self.target_net)

        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())

        self.a_zeros = torch.zeros(1, 1).long().to(self.device)

        self.pi_rand = np.ones(self.action_space) / self.action_space
        self.pi_rand_batch = torch.FloatTensor(self.pi_rand).unsqueeze(0).repeat(self.batch, 1).to(self.device)

        self.q_loss = nn.SmoothL1Loss(reduction='none')

        if player:

            # play variables
            self.env = Env()
            self.trajectory = []
            self.images = []
            self.choices = np.arange(self.action_space, dtype=np.int)
            self.n_replay_saved = 1
            self.frame = 0
            self.states = 0

        else:

            # datasets
            self.train_dataset = Memory(root_dir)
            self.train_sampler = ReplayBatchSampler(root_dir)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            collate_fn=collate, num_workers=args.cpu_workers,
                                                            pin_memory=True, drop_last=False)

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        if torch.cuda.device_count() > 1:
            state = {'value_net': self.value_net.module.state_dict(),
                     'target_net': self.target_net.module.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'aux': aux}
        else:
            state = {'value_net': self.value_net.state_dict(),
                     'target_net': self.target_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        if torch.cuda.device_count() > 1:
            self.value_net.module.load_state_dict(state['value_net'])
            self.target_net.module.load_state_dict(state['target_net'])
        else:
            self.value_net.load_state_dict(state['value_net'])
            self.target_net.load_state_dict(state['target_net'])

        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def learn(self, n_interval, n_tot):

        self.value_net.train()
        self.target_net.eval()

        results = {'n': [], 'loss_value': [], 'loss_beta': [], 'act_diff': [], 'a_agent': [],
                   'a_player': [], 'loss_std': [], 'mc_val': [], "Hbeta": [], "Hpi": [],
                   "adv_a": [], "q_a": [], 'image': []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            n = self.n_offset + n + 1

            s = sample['s'].to(self.device)
            a = sample['a'].to(self.device).unsqueeze_(1)
            r = sample['r'].to(self.device)
            t = sample['t'].to(self.device)
            s_tag = sample['s_tag'].to(self.device)
            tde = sample['tde'].to(self.device)

            _, _, _, q_tag_eval, _ = self.value_net(s_tag, a, self.pi_rand_batch)
            q_tag_eval = q_tag_eval.detach()

            _, _, _, q_tag_target, _ = self.target_net(s_tag, a, self.pi_rand_batch)
            q_tag_target = q_tag_target.detach()

            _, _, _, q, q_a = self.value_net(s, a, self.pi_rand_batch)

            a_tag = torch.argmax(q_tag_eval, dim=1).unsqueeze(1)
            r = h_torch(r + self.discount ** self.n_steps * (1 - t) * hinv_torch(q_tag_target.gather(1, a_tag).squeeze(1)))

            is_value = tde ** (-self.priority_beta)
            is_value = is_value / is_value.max()
            loss_value = (self.q_loss(q_a, r) * is_value).mean()

            # Learning part

            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

            # collect actions statistics

            if not n % 50:

                a_index_np = a.squeeze(1).data.cpu().numpy()

                q_a = q_a.data.cpu().numpy()
                r = r.data.cpu().numpy()

                _, beta_index = q.data.cpu().max(1)
                beta_index = beta_index.numpy()
                act_diff = (a_index_np != beta_index).astype(np.int)

                # add results

                results['act_diff'].append(act_diff)
                results['a_agent'].append(beta_index)
                results['adv_a'].append(q_a)
                results['q_a'].append(q_a)
                results['a_player'].append(a_index_np)
                results['Hbeta'].append(0)
                results['Hpi'].append(0)
                results['mc_val'].append(r)

                # add results
                results['loss_beta'].append(0)
                results['loss_value'].append(loss_value.data.cpu().numpy())
                results['loss_std'].append(0)
                results['n'].append(n)

                if not n % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': self.n_offset + n + 1})

                if not n % self.update_target_interval:
                    # save agent state
                    self.target_net.load_state_dict(self.value_net.state_dict())

                if not n % n_interval:
                    results['act_diff'] = np.concatenate(results['act_diff'])
                    results['a_agent'] = np.concatenate(results['a_agent'])
                    results['adv_a'] = np.concatenate(results['adv_a'])
                    results['q_a'] = np.concatenate(results['q_a'])
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['mc_val'] = np.concatenate(results['mc_val'])
                    results['image'] = s[0, :-1, :, :].data.cpu()

                    yield results
                    self.value_net.train()
                    results = {key: [] for key in results}

                    if n >= n_tot:
                        break

    def play(self, n_tot, save=True, load=True, fix=False):

        pi_rand = np.ones(self.action_space) / self.action_space
        pi_rand = torch.FloatTensor(pi_rand).unsqueeze(0).to(self.device)

        for i in range(n_tot):

            self.env.reset()

            rewards = [[]]
            q_val = []
            lives = self.env.lives

            while not fix:
                try:
                    self.load_checkpoint(self.snapshot_path)
                    break
                except:
                    time.sleep(0.5)

            self.value_net.eval()

            while not self.env.t:

                if load and not (self.states % self.load_memory_interval):
                    try:
                        self.load_checkpoint(self.snapshot_path)
                    except:
                        pass

                    self.value_net.eval()

                s = self.env.s.to(self.device)
                # get aux data
                _, _, _, q, _ = self.value_net(s, self.a_zeros, pi_rand)

                q = q.squeeze(0).data.cpu().numpy()

                if self.n_offset >= self.random_initialization:

                    pi = np.zeros(self.action_space)
                    pi[np.argmax(q)] = 1

                else:

                    pi = self.pi_rand

                pi_mix = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi

                pi_mix = pi_mix.clip(0, 1)
                pi_mix = pi_mix / pi_mix.sum()
                a = np.random.choice(self.choices, p=pi_mix)
                self.env.step(a)

                if self.env.k >= self.history_length:

                    if lives > self.env.lives:
                        rewards.append([])
                    lives = self.env.lives

                    rewards[-1].append(self.env.r)
                    q_val.append(q[a])

                self.frame += 1

            mc_val = get_mc_value(rewards, None, self.discount, None)
            q_val = np.array(q_val)

            yield {'score': self.env.score,
                   'frames': self.env.k, "n": self.n_offset, "mc": mc_val, "q": q_val}

            if self.n_offset >= self.n_tot and not fix:
                break

        raise StopIteration

    def stm_fetch_gate(self, s):

        self.policy_stm.train()
        self.memory_stm.train()
        a_hat = self.policy_stm(s)
        a_hat = a_hat.detach()
        a_hat = torch.argmax(a_hat, dim=1)

        mean, _, _ = self.memory_stm(s)
        mean = mean.detach()

        z = (a_hat == torch.argmax(mean, dim=1))
        z = (z * (mean > 0.1)).long()

        a_hat = a_hat * z - (1 - z)

        return a_hat

    def stm_learn_gate(self, q_a, rr, v_0, v_n, tde_quantiles):

        r = np.sum(self.discount ** np.arange(self.n_steps) * rr)
        tde = r + self.discount ** self.n_steps * v_n - q_a

        return (np.sign(tde) * np.abs(tde / v_0)) > tde_quantiles

    def stm_learn_routine(self, s, a):

        s = torch.stack(s)
        a = torch.stack(a)

        s_gen = torch.sigmoid(self.memory_stm.gen_samples(self.stm_batch, source='learn').detach()).view(s.shape)
        theta_snap_net = list(map(lambda x: x.clone().detach(), self.policy_stm.parameters()))
        theta_snap_mem = list(map(lambda x: x.clone().detach(), self.memory_stm.parameters()))

        # get punctuation pattern
        masks = list(map(lambda x: x[0].bernoulli_(p=self.punctuation if x[1] else 0),
                         zip(self.net_masks, self.net_punct_list)))

        with torch.no_grad():
            for param, mask, org in zip(self.policy_stm.parameters(), masks, self.net_state_0):
                param.data = mask * org.data + (1 - mask) * param

        # initialize augmented lagrange parameter
        mu = torch.cuda.FloatTensor(1).fill_(1)
        lagrange = torch.cuda.FloatTensor(1).fill_(1)

        for _ in range(self.n_iter_net):
            theta_current = list(self.policy_stm.parameters())

            theta_dist = 0.5 * torch.stack([((1 - mask) * (theta_i - theta_j) ** 2).sum() for theta_i, theta_j, mask
                                            in zip(theta_snap_net, theta_current, masks)]).sum()

            a_est = self.policy_stm(s)
            objective = self.loss_classification(a_est, a)

            # PA
            loss_classification = theta_dist + mu / 2 * objective ** 2 + lagrange * objective

            self.optimizer_net.zero_grad()
            loss_classification.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_stm.parameters(), 1000, norm_type=2)
            self.optimizer_net.step()

            lagrange += mu * objective.detach()

        # NOW FOR THE MEMORY TRAINING

        # get punctuation pattern
        masks = list(map(lambda x: x[0].bernoulli_(p=self.punctuation if x[1] else 0),
                         zip(self.mem_masks, self.mem_punct_list)))

        with torch.no_grad():
            for param, mask, org in zip(self.memory_stm.parameters(), masks, self.mem_state_0):
                param.data = mask * org.data + (1 - mask) * param

        # initialize augmented lagrange parameter
        mu = torch.cuda.FloatTensor(1).fill_(1)
        lagrange = torch.cuda.FloatTensor(1).fill_(1)

        for _ in range(self.n_iter_mem):
            theta_current = list(self.memory_stm.parameters())

            theta_dist = 0.5 * torch.stack([((1 - mask) * (theta_i - theta_j) ** 2).sum() for theta_i, theta_j, mask
                                            in zip(theta_snap_mem, theta_current, masks)]).sum()

            s_reconstruct, _, _, kl = self.memory_stm(s, a)
            s_reconstruct = s_reconstruct.view(*s.shape)

            objective = (self.loss_reconstruction(s_reconstruct, s).mean(dim=0).sum() + kl.mean())

            _, _, kl_forget = self.memory_stm(s_gen)

            objective += kl_forget.mean()

            # PA
            loss_reconstruction = theta_dist + mu / 2 * objective ** 2 + lagrange * objective

            self.optimizer_mem.zero_grad()
            loss_reconstruction.backward()
            torch.nn.utils.clip_grad_norm_(self.memory_stm.parameters(), 1000, norm_type=2)
            self.optimizer_mem.step()

            lagrange += mu * objective.detach()

    def multiplay(self):

        n_players = self.n_players

        player_i = np.arange(self.actor_index, self.actor_index + self.n_actors * n_players, self.n_actors) / (self.n_actors * n_players - 1)
        mp_explore = 0.4 ** (1 + 7 * player_i)
        explore_threshold = player_i

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0
        episode_num = np.zeros(n_players, dtype=np.int)
        a_zeros_mp = self.a_zeros.repeat(n_players, 1)
        mp_pi_rand = np.repeat(np.expand_dims(self.pi_rand, axis=0), n_players, axis=0)
        range_players = np.arange(n_players)
        rewards = [[[]] for _ in range(n_players)]
        v_target = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        q_a = [[] for _ in range(n_players)]
        image_dir = ['' for _ in range(n_players)]
        trajectory = [[] for _ in range(n_players)]
        tde_track = [np.zeros(1) for _ in range(n_players)]
        stm_queue = []
        stm_action_queue = []
        s_fifo = [None] * self.n_steps
        tde_quantile = 0

        screen_dir = [os.path.join(self.explore_dir, "screen")] * n_players
        trajectory_dir = [os.path.join(self.explore_dir, "trajectory")] * n_players
        readlock = [os.path.join(self.list_dir, "readlock_explore.npy")] * n_players

        pi_rand_batch = torch.FloatTensor(self.pi_rand).unsqueeze(0).repeat(n_players, 1).to(self.device)

        for i in range(n_players):
            mp_env[i].reset()

            # set initial episodes number
            # lock read
            fwrite = lock_file(self.episodelock)
            episode_num[i] = np.load(fwrite).item()
            fwrite.seek(0)
            np.save(fwrite, episode_num[i] + 1)
            # unlock file
            release_file(fwrite)

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

            s = torch.cat([env.s for env in mp_env]).to(self.device)
            a_stm = self.stm_fetch_gate(s)

            _, _, _, q, _ = self.value_net(s, a_zeros_mp, pi_rand_batch)

            q = q.data.cpu().numpy()

            mp_trigger = np.logical_and(
                np.array([env.score for env in mp_env]) >= self.behavioral_avg_score * explore_threshold,
                explore_threshold >= 0)
            exploration = np.repeat(np.expand_dims(mp_explore * mp_trigger, axis=1), self.action_space, axis=1)

            if self.n_offset >= self.random_initialization:

                pi = np.zeros((self.n_players, self.action_space))
                pi[range_players, np.argmax(q, axis=1)] = 1

            else:
                pi = mp_pi_rand

            pi_mix = pi * (1 - exploration) + exploration * mp_pi_rand
            pi_mix = pi_mix.clip(0, 1)
            pi_mix = pi_mix / np.repeat(np.expand_dims(pi_mix.sum(axis=1), axis=1), self.action_space, axis=1)

            v_expected = (q * pi).sum(axis=1)

            for i in range(n_players):

                a = np.random.choice(self.choices, p=pi_mix[i])

                env = mp_env[i]
                cv2.imwrite(os.path.join(image_dir[i], "%s.png" % str(self.frame)), mp_env[i].image, [imcompress, compress_level])

                episode[i].append(np.array((self.frame, a, pi[i],
                                            None, None,
                                            episode_num[i], 0., 0, 0,
                                            0., 1., 1., 0, 1., 0), dtype=self.rec_type))

                a = a if a_stm[i] < 0 else a_stm[i]

                env.step(a)

                if lives[i] > env.lives:
                    rewards[i].append([])
                    v_target[i].append([])
                lives[i] = env.lives

                rewards[i][-1].append(env.r)
                v_target[i][-1].append(v_expected[i])
                q_a[i].append(q[i][a])

                if self.stm_learn_gate(q_a[i][-self.n_steps-1], rewards[i][-self.n_steps-1:-2], v_target[i][-self.n_steps],
                                       v_target[i][-1], tde_quantile):
                    stm_queue.append(s[i])
                    stm_action_queue.append(a)
                    if len(stm_queue) == self.stm_batch:
                        self.stm_learn_routine(stm_queue, stm_action_queue)
                        stm_queue = []
                        stm_action_queue = []

                # add to fifo
                s_fifo.pop(0)
                s_fifo.append(s)

                if env.t:

                    td_val, t_val = get_tde_value(rewards[i], self.discount, self.n_steps)
                    tde = np.array(q_a[i]) - get_td_value(rewards[i], v_target[i], self.discount, self.n_steps)

                    v_scale = np.concatenate(v_target[i])
                    # compute a signed version of tde
                    tde = np.sign(tde) * np.minimum(np.abs(tde), np.abs(tde / v_scale))

                    # add
                    tde_track[i] = tde
                    tde_quantile = np.quantile(np.concatenate(tde_track), 0.9)

                    # add absolute value and squashing
                    tde = np.abs(tde) ** self.priority_alpha

                    episode_df = np.stack(episode[i][self.history_length - 1:self.max_length])
                    episode_df['r'] = td_val[self.history_length - 1:self.max_length]
                    episode_df['t'] = t_val[self.history_length - 1:self.max_length]
                    episode_df['tde'] = tde[self.history_length - 1:self.max_length]

                    trajectory[i].append(episode_df)

                    print("ape | st: %d\t| sc: %d\t| f: %d\t| e: %7g\t| typ: %2d | trg: %d | t: %d\t| n %d\t| avg_r: %g\t| avg_f: %g" %
                        (self.frame, env.score, env.k, mp_explore[i], np.sign(explore_threshold[i]), mp_trigger[i],
                         time.time() - self.start_time, self.n_offset, self.behavioral_avg_score,
                         self.behavioral_avg_frame))

                    env.reset()
                    episode[i] = []
                    q_a[i] = []
                    rewards[i] = [[]]
                    v_target[i] = [[]]
                    lives[i] = env.lives

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

            self.frame += 1
            if not self.frame % self.player_replay_size:
                yield True

            if self.n_offset >= self.n_tot:
                break

    def demonstrate(self, n_tot):

        self.value_net.eval()
        pi_rand = np.ones(self.action_space) / self.action_space
        pi_rand = torch.FloatTensor(pi_rand).unsqueeze(0).to(self.device)

        for i in range(n_tot):

            self.env.reset()

            choices = np.arange(self.action_space, dtype=np.int)

            while not self.env.t:

                s = self.env.s.to(self.device)
                _, _, _, q, _ = self.value_net(s, self.a_zeros, pi_rand)

                q = q.squeeze(0).data.cpu().numpy()

                pi_greed = np.zeros(self.action_space)
                pi_greed[np.argmax(q)] = 1
                pi = self.epsilon * self.pi_rand + (1 - self.epsilon) * pi_greed

                pi = pi.clip(0, 1)
                pi = pi / pi.sum()

                a = np.random.choice(choices, p=pi)
                self.env.step(a)

                img = state_to_img(s)

                v = (pi * q).sum()
                adv = q - v

                yield {'score': self.env.score, "beta": pi, "v": v, "q": q, "adv": adv, "s": img, 'frames': self.env.k,
                       "actions": self.env.action_meanings, "a": a
                       }

        raise StopIteration
