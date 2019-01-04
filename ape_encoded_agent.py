import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from comet_ml import Experiment as comet
import psutil

from config import consts, args

from model import InverseDynamics, StateEncoder, DQNTruncated

from memory import DQNMemory, DQNBatchSampler
from agent import Agent
from environment import Env
from preprocess import get_tde_value, lock_file, release_file, h_torch, hinv_torch, _get_mc_value
import cv2
import os
import time
import shutil
import itertools

imcompress = cv2.IMWRITE_PNG_COMPRESSION
compress_level = 2
mem_threshold = consts.mem_threshold

class ApeAgent(Agent):

    def __init__(self, root_dir, player=False, choose=False, checkpoint=None):

        print("Learning with Ape Encoded Agent")
        super(ApeAgent, self).__init__()
        self.checkpoint = checkpoint
        self.root_dir = root_dir
        self.best_player_dir = os.path.join(root_dir, "best")
        self.snapshot_path = os.path.join(root_dir, "snapshot")
        self.exploit_dir = os.path.join(root_dir, "exploit")
        self.explore_dir = os.path.join(root_dir, "explore")
        self.list_dir = os.path.join(root_dir, "list")
        self.writelock = os.path.join(self.list_dir, "writelock.npy")
        self.episodelock = os.path.join(self.list_dir, "episodelock.npy")

        self.device = torch.device("cuda:%d" % self.cuda_id)

        self.dqn_net = DQNTruncated().to(self.device)
        self.state_encoder = StateEncoder().to(self.device)
        self.inverse_dynamics = InverseDynamics().to(self.device)

        self.a_zeros = torch.zeros(1, 1).long().to(self.device)
        self.a_zeros_batch = self.a_zeros.repeat(self.batch, 1)

        self.pi_rand = np.ones(self.action_space) / self.action_space

        if player:

            # play variables
            self.env = Env()
            self.trajectory = []
            self.images = []
            self.choices = np.arange(self.action_space, dtype=np.int)
            self.pi_rand = np.ones(self.action_space) / self.action_space
            self.n_replay_saved = 1
            self.frame = 0
            self.states = 0

            print("Explorer player")
            self.trajectory_dir = os.path.join(self.explore_dir, "trajectory")
            self.screen_dir = os.path.join(self.explore_dir, "screen")
            self.readlock = os.path.join(self.list_dir, "readlock_explore.npy")

        else:

            # datasets
            self.train_dataset = DQNMemory(root_dir)
            self.train_sampler = DQNBatchSampler(root_dir)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_sampler,
                                                            num_workers=args.cpu_workers, pin_memory=True, drop_last=False)
            try:
                os.mkdir(self.best_player_dir)
                os.mkdir(self.exploit_dir)
                os.mkdir(self.explore_dir)
                os.mkdir(os.path.join(self.exploit_dir, "trajectory"))
                os.mkdir(os.path.join(self.exploit_dir, "screen"))
                os.mkdir(os.path.join(self.explore_dir, "trajectory"))
                os.mkdir(os.path.join(self.explore_dir, "screen"))
                os.mkdir(self.list_dir)
                np.save(self.writelock, 0)
                np.save(self.episodelock, 0)
                np.save(os.path.join(self.list_dir, "readlock_explore.npy"), [])
                np.save(os.path.join(self.list_dir, "readlock_exploit.npy"), [])
            except FileExistsError:
                pass

        # configure learning

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        self.optimizer = torch.optim.Adam(self.dqn_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.optimizer_state = torch.optim.Adam(itertools.chain(self.state_encoder.parameters(), self.inverse_dynamics.parameters()),
                                                          lr=0.00025 / 4, eps=1.5e-4, weight_decay=0)

        self.loss_dynamics = torch.nn.CrossEntropyLoss()
        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        state = {'dqn_net': self.dqn_net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'optimizer_state': self.optimizer_state.state_dict(),
                 'state_encoder': self.state_encoder.state_dict(),
                 'inverse_dynamics': self.inverse_dynamics.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)
        self.dqn_net.load_state_dict(state['dqn_net'])
        self.state_encoder.load_state_dict(state['state_encoder'])
        self.inverse_dynamics.load_state_dict(state['inverse_dynamics'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.optimizer_state.load_state_dict(state['optimizer_state'])
        self.n_offset = state['aux']['n']

        return state['aux']

    def resume(self, model_path):
        aux = self.load_checkpoint(model_path)
        return aux

    def learn(self, n_interval, n_tot):

        self.dqn_net.train()
        self.inverse_dynamics.train()
        self.state_encoder.train()

        target_net = DQNTruncated().to(self.device)
        target_net.load_state_dict(self.dqn_net.state_dict())
        target_net.eval()

        results = {'n': [], 'loss_value': [], 'loss_beta': [], 'act_diff': [], 'a_agent': [],
                   'a_player': [], 'loss_std': [], 'mc_val': [], "Hbeta": [], "Hpi": [], "adv_a": [], "q_a": []}

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = sample['s'].to(self.device)
            a = sample['a'].to(self.device).unsqueeze_(1)
            r = sample['r'].to(self.device)
            aux = sample['aux'].to(self.device)

            t = sample['t'].to(self.device)
            s_tag = sample['s_tag'].to(self.device)
            aux_tag = sample['aux_tag'].to(self.device)

            f = self.state_encoder(s, aux)
            f_tag = self.state_encoder(s_tag, aux_tag)

            s = f.detach()
            s_tag = f_tag.detach()

            _, _, _, q_tag_target, _ = target_net(s_tag, self.a_zeros_batch)
            q_tag_target = q_tag_target.detach()

            _, _, _, q_tag_eval, _ = self.dqn_net(s_tag, self.a_zeros_batch)
            q_tag_eval = q_tag_eval.detach()

            _, _, _, q, q_a = self.dqn_net(s, a)
            q_a_eval = q_a.detach()

            a_tag = torch.argmax(q_tag_eval, dim=1).unsqueeze(1)
            r = h_torch(r + self.discount ** self.n_steps * (1 - t) * hinv_torch(q_tag_target.gather(1, a_tag).squeeze(1)))

            is_value = ((r - q_a_eval).abs() + 0.001) ** self.priority_beta
            is_value = is_value / is_value.mean()
            loss_value = ((q_a - r) ** 2 * is_value).mean()

            # dynamics loss

            p_a = self.inverse_dynamics(f, f_tag)
            loss_dynamics = self.loss_dynamics(p_a, a.squeeze(1))

            # Learning part

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            self.optimizer_state.zero_grad()
            loss_dynamics.backward()
            self.optimizer_state.step()

            # collect actions statistics

            if not (n + 1) % 50:

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

                if not (n+1) % self.update_memory_interval:
                    # save agent state
                    self.save_checkpoint(self.snapshot_path, {'n': self.n_offset + n + 1})

                if not (n+1) % self.update_target_interval:
                    # save agent state
                    target_net.load_state_dict(self.dqn_net.state_dict())

                if not (n+1) % n_interval:
                    results['act_diff'] = np.concatenate(results['act_diff'])
                    results['a_agent'] = np.concatenate(results['a_agent'])
                    results['adv_a'] = np.concatenate(results['adv_a'])
                    results['q_a'] = np.concatenate(results['q_a'])
                    results['a_player'] = np.concatenate(results['a_player'])
                    results['mc_val'] = np.concatenate(results['mc_val'])

                    yield results
                    self.dqn_net.train()
                    results = {key: [] for key in results}

                    if (n + self.n_offset) >= n_tot:
                        break

    def clean(self):

        while True:

            time.sleep(2)

            screen_dir = os.path.join(self.explore_dir, "screen")
            trajectory_dir = os.path.join(self.explore_dir, "trajectory")

            try:
                del_inf = np.load(os.path.join(self.list_dir, "old_explore.npy"))
            except (IOError, ValueError):
                continue
            traj_min = del_inf[0] - 32
            episode_list = set()

            for traj in os.listdir(trajectory_dir):
                traj_num = int(traj.split(".")[0])
                if traj_num < traj_min:
                    traj_data = np.load(os.path.join(trajectory_dir, traj))
                    for d in traj_data:
                        episode_list.add(d['ep'])
                    os.remove(os.path.join(trajectory_dir, traj))

            for ep in episode_list:
                shutil.rmtree(os.path.join(screen_dir, str(ep)))

    def play(self, n_tot, save=True, load=True, fix=False):

        for i in range(n_tot):

            self.env.reset()

            rewards = [[]]
            q_val = []
            lives = self.env.lives
            trigger = False

            while not fix:
                try:
                    self.load_checkpoint(self.snapshot_path)
                except:
                    time.sleep(0.5)
                    self.load_checkpoint(self.snapshot_path)

            self.dqn_net.eval()

            while not self.env.t:

                if load and not (self.states % self.load_memory_interval):
                    try:
                        self.load_checkpoint(self.snapshot_path)
                    except:
                        pass

                    self.dqn_net.eval()

                s = self.env.s.to(self.device)
                trigger = trigger or (self.env.score > self.behavioral_avg_score * self.explore_threshold)
                # get aux data
                aux = self.env.aux.to(self.device)

                s = self.state_encoder(s, aux)
                _, _, _, q, _ = self.dqn_net(s, self.a_zeros)

                q = q.squeeze(0).data.cpu().numpy()

                if self.n_offset >= self.random_initialization:

                    pi = np.zeros(self.action_space)
                    pi[np.argmax(q)] = 1

                else:
                    pi = self.pi_rand

                if self.off:

                    if trigger:
                        exploration = self.eps_post
                    else:
                        exploration = self.eps_pre

                    pi_mix = exploration * self.pi_rand + (1 - exploration) * pi

                else:
                    pi_mix = pi

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

            # tde_val, t_val = get_tde_value(rewards, self.discount, self.n_steps)

            mc_val = _get_mc_value(rewards, None, self.discount, None)
            q_val = np.array(q_val)

            # print("mc_val: %d | q_val: %d" % (len(mc_val), len(q_val)))

            yield {'score': self.env.score,
                   'frames': self.env.k, "n": self.n_offset, "mc": mc_val, "q": q_val}

            if self.n_offset >= self.n_tot and not fix:
                break

        raise StopIteration

    def multiplay(self):

        n_players = self.n_players

        player_i = np.arange(self.actor_index, self.actor_index + self.n_actors * n_players, self.n_actors) / (self.n_actors * n_players - 1)
        explore_threshold = np.zeros(self.n_players)
        mp_explore = 0.4 ** (1 + 7 * player_i)

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0
        episode_num = np.zeros(n_players, dtype=np.int)
        a_zeros_mp = self.a_zeros.repeat(n_players, 1)
        mp_pi_rand = np.repeat(np.expand_dims(self.pi_rand, axis=0), n_players, axis=0)
        range_players = np.arange(n_players)
        rewards = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
        image_dir = ['' for _ in range(n_players)]
        trajectory = [[] for _ in range(n_players)]

        screen_dir = [os.path.join(self.explore_dir, "screen")] * n_players
        trajectory_dir = [os.path.join(self.explore_dir, "trajectory")] * n_players
        readlock = [os.path.join(self.list_dir, "readlock_explore.npy")] * n_players

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

                self.dqn_net.eval()

            s = torch.cat([env.s for env in mp_env]).to(self.device)
            aux = torch.cat([env.aux for env in mp_env]).to(self.device)
            aux_np = aux.cpu().numpy().astype(np.float32)

            s = self.state_encoder(s, aux)
            _, _, _, q, _ = self.dqn_net(s, a_zeros_mp)

            q = q.data.cpu().numpy()

            if self.n_offset >= self.random_initialization:

                pi = np.zeros((self.n_players, self.action_space))
                pi[range_players, np.argmax(q, axis=1)] = 1

            else:
                pi = mp_pi_rand

            mp_trigger = np.array([env.score for env in mp_env]) >= self.behavioral_avg_score * explore_threshold
            exploration = np.repeat(np.expand_dims(mp_explore * mp_trigger + (1 - mp_trigger) * self.eps_pre, axis=1), self.action_space, axis=1)
            pi_mix = pi * (1 - exploration) + exploration * mp_pi_rand
            pi_mix = pi_mix.clip(0, 1)
            pi_mix = pi_mix / np.repeat(np.expand_dims(pi_mix.sum(axis=1), axis=1), self.action_space, axis=1)

            for i in range(n_players):

                a = np.random.choice(self.choices, p=pi_mix[i])

                env = mp_env[i]
                cv2.imwrite(os.path.join(image_dir[i], "%s.png" % str(self.frame)), mp_env[i].image, [imcompress, compress_level])
                episode[i].append({"fr": self.frame, "a": a, "r": 0, "aux": aux_np[i], "ep": episode_num[i], "t": 0})

                env.step(a)

                if lives[i] > env.lives:
                    rewards[i].append([])
                lives[i] = env.lives

                rewards[i][-1].append(env.r)

                if env.t:

                    td_val, t_val = get_tde_value(rewards[i], self.discount, self.n_steps)

                    for j, record in enumerate(episode[i]):
                        record['r'] = td_val[j]
                        record['t'] = t_val[j]

                    trajectory[i] += episode[i][self.history_length - 1:self.max_length]

                    print("rbi | st: %d | sc: %d | e: %g | n-step: %s | n %d | avg: %g" %
                          (self.frame, env.score, mp_explore[i], str(self.n_steps), self.n_offset,
                           self.behavioral_avg_score))

                    env.reset()
                    episode[i] = []
                    rewards[i] = [[]]
                    lives[i] = env.lives
                    mp_trigger[i] = 0

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

                    if len(trajectory[i]) >= self.player_replay_size:

                        # write if enough space is available
                        if psutil.virtual_memory().available >= mem_threshold:

                            # lock read
                            fwrite = lock_file(self.writelock)
                            traj_num = np.load(fwrite).item()
                            fwrite.seek(0)
                            np.save(fwrite, traj_num + 1)
                            # unlock file
                            release_file(fwrite)

                            traj_to_save = trajectory[i]

                            for j in range(len(traj_to_save)):
                                traj_to_save[j]['traj'] = traj_num

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

    def set_player(self, player, cmin=None, cmax=None, delta=None, eps_pre=None,
                   eps_post=None, temp_soft=None, behavioral_avg_score=None,
                   behavioral_avg_frame=None, explore_threshold=None):

        self.player = player

        if eps_pre is not None:
            self.eps_pre = eps_pre

        if eps_post is not None:
            self.eps_post = eps_post

        if temp_soft is not None:
            self.temp_soft = temp_soft


        if cmin is not None:
            self.cmin = cmin

        if cmax is not None:
            self.cmax = cmax

        if delta is not None:
            self.delta = delta

        if explore_threshold is not None:
            self.explore_threshold = explore_threshold

        if behavioral_avg_score is not None:
            self.behavioral_avg_score = behavioral_avg_score

        if behavioral_avg_frame is not None:
            self.behavioral_avg_frame = behavioral_avg_frame

        self.off = True if max(self.eps_post, self.eps_pre) > 0 else False

        if self.off:
            self.trajectory_dir = os.path.join(self.explore_dir, "trajectory")
            self.screen_dir = os.path.join(self.explore_dir, "screen")
        else:
            self.trajectory_dir = os.path.join(self.exploit_dir, "trajectory")
            self.screen_dir = os.path.join(self.exploit_dir, "screen")

    def demonstrate(self, n_tot):

        self.dqn_net.eval()

        for i in range(n_tot):

            self.env.reset()

            # here there is a problem when there is a varying/increasing life counter as in mspacman

            choices = np.arange(self.action_space, dtype=np.int)

            while not self.env.t:

                s = self.env.s.to(self.device)
                aux = self.env.aux.to(self.device)

                v, adv, _, q, _ = self.dqn_net(s, self.a_zeros, aux)
                v = v.squeeze(0)
                adv = adv.squeeze(0)
                q = q.squeeze(0).data.cpu().numpy()
                beta = np.zeros(self.action_space)
                beta[np.argmax(q)] = 1

                pi = beta * (1 - 0.00164) + 0.00164 * self.pi_rand

                pi = pi.clip(0, 1)
                pi = pi / pi.sum()

                a = np.random.choice(choices, p=pi)
                self.env.step(a)

                yield {'score': self.env.score,
                       "beta": beta,
                       "v": v.data.cpu().numpy(),
                       "q": q,
                       "aux": aux.squeeze(0).data.cpu().numpy(),
                       "adv": adv.data.cpu().numpy(),
                       "o": s.squeeze(0).data[:3].cpu().numpy(),
                       'frames': self.env.k,
                       "actions": self.env.action_meanings,
                       }

        raise StopIteration