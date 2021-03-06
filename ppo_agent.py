import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

from config import consts, args
import psutil
import socket

from model import BehavioralNet, ValueNet

from memory import ObservationsMemory, ObservationsBatchSampler
from agent import Agent
from environment import Env
from preprocess import _get_mc_value, get_expected_value, release_file, lock_file, get_gae_est
import cv2
import os
import time
import shutil

imcompress = cv2.IMWRITE_PNG_COMPRESSION
compress_level = 2
mem_threshold = consts.mem_threshold


class PPOAgent(Agent):

    def __init__(self, root_dir, player=False, choose=False, checkpoint=None):

        print("Learning with RBIAgent")
        super(PPOAgent, self).__init__(root_dir, checkpoint)

        self.beta_net = BehavioralNet().to(self.device)
        self.value_net = ValueNet().to(self.device)

        self.pi_rand = np.ones(self.action_space) / self.action_space
        self.pi_rand_batch = torch.FloatTensor(self.pi_rand).unsqueeze(0).repeat(self.batch, 1).to(self.device)

        self.a_zeros = torch.zeros(1, 1).long().to(self.device)
        self.a_zeros_batch = self.a_zeros.repeat(self.batch, 1)

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

            print("Explorer player")
            self.trajectory_dir = os.path.join(self.explore_dir, "trajectory")
            self.screen_dir = os.path.join(self.explore_dir, "screen")
            self.readlock = os.path.join(self.list_dir, "readlock_explore.npy")

        else:

            # self.target_net = DuelNet().to(self.device)
            # datasets
            self.train_dataset = ObservationsMemory(root_dir)
            self.train_sampler = ObservationsBatchSampler(root_dir)
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
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)
        self.optimizer_beta = torch.optim.Adam(self.beta_net.parameters(), lr=0.00025/4, eps=1.5e-4, weight_decay=0)

        self.n_offset = 0

    def save_checkpoint(self, path, aux=None):

        state = {'beta_net': self.beta_net.state_dict(),
                 'value_net': self.value_net.state_dict(),
                 'optimizer_value': self.optimizer_value.state_dict(),
                 'optimizer_beta': self.optimizer_beta.state_dict(),
                 'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        self.beta_net.load_state_dict(state['beta_net'])
        self.value_net.load_state_dict(state['value_net'])
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

        results = {'n': [], 'loss_value': [], 'loss_beta': [], 'act_diff': [], 'a_agent': [],
                   'a_player': [], 'loss_std': [], 'mc_val': [], "Hbeta": [], "Hpi": [], "adv_a": [], "q_a": []}

        # tic = time.time()

        for n, sample in tqdm(enumerate(self.train_loader)):

            s = sample['s'].to(self.device)
            a = sample['a'].to(self.device).unsqueeze_(1)
            r = sample['r'].to(self.device)
            rho = sample['rho'].to(self.device)
            pi = sample['pi'].to(self.device)
            aux = sample['aux'].to(self.device)

            # Behavioral nets
            beta = self.beta_net(s, aux)
            beta_log = F.log_softmax(beta, dim=1)
            beta_soft = F.softmax(beta, dim=1)

            v = self.value_net(s, aux)

            v = v.squeeze(1)
            loss_value = ((v - r) ** 2).mean()

            rt = (beta_soft / pi).gather(1, a)
            ppo_objective = torch.min(rt * rho, rho * torch.clamp(rt, min=1 - self.ppo_eps, max=1 + self.ppo_eps))

            Hbeta = -(beta_soft * beta_log).sum(dim=1)

            loss_beta = - ppo_objective.mean() - 0.01 * Hbeta.mean()            # Learning part

            self.optimizer_beta.zero_grad()
            loss_beta.backward()
            self.optimizer_beta.step()

            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

            # collect actions statistics

            if not (n + 1 + self.n_offset) % 50:

                a_index_np = a[:, 0].data.cpu().numpy()

                # avoid zero pi
                pi = pi.clamp(min=1e-4, max=1)
                pi /= pi.sum(dim=1).unsqueeze(1).repeat(1, self.action_space)

                pi_log = pi.log()
                beta_soft = F.softmax(beta, dim=1).detach()

                Hpi = -(pi * pi_log).sum(dim=1)
                Hbeta = -(beta_soft * beta_log).sum(dim=1)

                adv_a = np.ones(self.batch)
                q_a = np.ones(self.batch)
                r = r.data.cpu().numpy()

                _, beta_index = beta_soft.data.cpu().max(1)
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
                results['loss_std'].append(0)
                results['n'].append(n)

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

                    yield results
                    self.beta_net.train()
                    self.value_net.train()
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
            v_target = [[]]
            q_val = []
            lives = self.env.lives
            trigger = False

            while not fix:
                try:
                    self.load_checkpoint(self.snapshot_path)
                    break
                except:
                    time.sleep(0.5)

            self.beta_net.eval()
            self.value_net.eval()

            while not self.env.t:

                if load and not (self.states % self.load_memory_interval):
                    try:
                        self.load_checkpoint(self.snapshot_path)
                    except:
                        pass

                    self.beta_net.eval()
                    self.value_net.eval()

                s = self.env.s.to(self.device)
                trigger = trigger or (self.env.score > self.behavioral_avg_score * self.explore_threshold)
                # get aux data
                aux = self.env.aux.to(self.device)

                beta = self.beta_net(s, aux)
                beta = F.softmax(beta.detach(), dim=1)

                # take q as adv

                v = self.value_net(s, aux)

                v_expected = v.squeeze(0).data.cpu().numpy()
                beta = beta.squeeze(0).data.cpu().numpy()

                if self.n_offset >= self.random_initialization:

                    if self.player == "reroutetv":

                        pi_mix = beta.copy()

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
                    q_val.append(0)

                self.frame += 1

            mc_val = _get_mc_value(rewards, None, self.discount, None)
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

        player_i = np.arange(self.actor_index, self.actor_index + self.n_actors * n_players, self.n_actors) / (self.n_actors * n_players - 1)

        explore_threshold = player_i

        mp_explore = 0.4 ** (1 + 7 * (1 - player_i))

        mp_env = [Env() for _ in range(n_players)]
        self.frame = 0

        a_zeros_mp = self.a_zeros.repeat(n_players, 1)
        mp_pi_rand = np.repeat(np.expand_dims(self.pi_rand, axis=0), n_players, axis=0)

        range_players = np.arange(n_players)
        rewards = [[[]] for _ in range(n_players)]
        v_target = [[[]] for _ in range(n_players)]
        rho = [[[]] for _ in range(n_players)]
        episode = [[] for _ in range(n_players)]
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

            s = torch.cat([env.s for env in mp_env]).to(self.device)
            aux = torch.cat([env.aux for env in mp_env]).to(self.device)
            aux_np = aux.cpu().numpy().astype(np.float32)

            beta = self.beta_net(s, aux)
            beta = F.softmax(beta.detach(), dim=1)

            v_expected = self.value_net(s, aux)

            beta = beta.data.cpu().numpy()
            v_expected = v_expected.squeeze(1).data.cpu().numpy()

            mp_trigger = np.logical_and(
                np.array([env.score for env in mp_env]) >= self.behavioral_avg_score * explore_threshold,
                explore_threshold >= 0)

            if self.n_offset >= self.random_initialization:

                pi = beta.copy()
                # const explore
                pi_mix = pi.copy()

            else:
                pi = mp_pi_rand
                pi_mix = pi

            pi = pi.astype(np.float32)
            for i in range(n_players):

                a = np.random.choice(self.choices, p=pi_mix[i])

                env = mp_env[i]
                cv2.imwrite(os.path.join(image_dir[i], "%s.png" % str(self.frame)), mp_env[i].image, [imcompress, compress_level])
                episode[i].append({"fr": self.frame, "a": a, "r": 0, "pi": pi[i], "rho": 0,
                                   "aux": aux_np[i], "ep": episode_num[i], "t": 0})

                env.step(a)

                if lives[i] > env.lives:
                    rewards[i].append([])
                    v_target[i].append([])
                    rho[i].append([])
                lives[i] = env.lives

                rewards[i][-1].append(env.r)
                v_target[i][-1].append(v_expected[i])
                rho[i][-1].append(pi[i][a] / pi_mix[i][a])

                if env.t:

                    # cancel termination reward
                    rewards[i][-1][-1] -= self.termination_reward * int(env.k * self.skip >= self.max_length or env.score >= self.max_score)
                    td_val = get_expected_value(rewards[i], v_target[i], self.discount, self.n_steps)
                    rho_val = get_gae_est(rewards[i], v_target[i], self.discount)

                    rho_vec = np.concatenate(rho[i])
                    for j, record in enumerate(episode[i]):
                        record['r'] = td_val[j]
                        # record['rho'] = rho_val[j]
                        record['rho'] = np.array([rho_vec[j], rho_val[j]], dtype=np.float32)

                    trajectory[i] += episode[i][self.history_length-1:self.max_length]

                    print("rbi | st: %d\t| sc: %d\t| f: %d\t| e: %7g\t| typ: %2d | trg: %d | t: %d\t| n %d\t| avg_r: %g\t| avg_f: %g" %
                          (self.frame, env.score, env.k, mp_explore[i], np.sign(explore_threshold[i]), mp_trigger[i], time.time() - self.start_time, self.n_offset, self.behavioral_avg_score, self.behavioral_avg_frame))

                    env.reset()
                    episode[i] = []
                    rewards[i] = [[]]
                    v_target[i] = [[]]
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

            # write trajectory to dir

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
            self.eps_pre = eps_pre * self.action_space / (self.action_space - 1)

        if eps_post is not None:
            self.eps_post = eps_post * self.action_space / (self.action_space - 1)

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

        self.trajectory_dir = os.path.join(self.explore_dir, "trajectory")
        self.screen_dir = os.path.join(self.explore_dir, "screen")
        self.readlock = os.path.join(self.list_dir, "readlock_explore.npy")


    def demonstrate(self, n_tot):

        self.beta_net.eval()
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

                beta_softmax = F.softmax(beta, dim=1)

                v, adv, _, q, _ = self.value_net(s, self.a_zeros, beta_softmax, aux)
                v = v.squeeze(0)
                adv = adv.squeeze(0)
                q = q.squeeze(0).data.cpu().numpy()

                beta = beta.squeeze(0)
                beta = F.softmax(beta, dim=0)

                beta = torch.clamp((beta - self.entropy_loss / self.action_space) / (1 - self.entropy_loss),
                                   self.eps_pre / self.action_space, 1 - self.eps_pre)

                beta = beta / beta.sum()
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