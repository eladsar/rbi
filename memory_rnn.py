import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args
from preprocess import get_mc_value, hinv_np, lock_file, release_file
import itertools
import cv2
import pandas as pd


img_width = args.width
img_height = args.height
interpolation = cv2.INTER_LINEAR  # cv2.INTER_AREA  #
imread_grayscale = cv2.IMREAD_GRAYSCALE


class MemoryRNN(torch.utils.data.Dataset):

    def __init__(self):
        super(MemoryRNN, self).__init__()
        self.history_length = args.history_length
        self.n_steps = args.n_steps
        self.seq_length = args.seq_length
        self.burn_in = args.burn_in
        self.history_mat = np.expand_dims(np.arange(self.seq_length + self.burn_in), axis=1) + np.arange(self.history_length)

    def __len__(self):
        return args.n_tot

    def __getitem__(self, index):
        raise NotImplementedError

    def preprocess_trajectory(self, episode_dir, frame, k):

        frames = [os.path.join(episode_dir, "%d.png" % max(frame + i, -1))
                  for i in range(-self.history_length + 1, k)]
        # try:
        imgs = np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32),
                                         (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frames], axis=0)
        # except:
        #     print("XXX")
        return imgs[self.history_mat[:k], :, :]


class ObservationsRNNMemory(MemoryRNN):

    def __init__(self, replay_dir):

        super(ObservationsRNNMemory, self).__init__()
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    def __getitem__(self, samples):

        l = min(samples[self.burn_in]['fr'] - samples[self.burn_in]['fr_s'], self.burn_in)
        pad_l = self.burn_in - l
        k = min(samples[self.burn_in]['fr_e'] - samples[self.burn_in]['fr'], self.seq_length) + l
        samples = samples[self.burn_in-l:self.burn_in-l+k]
        pad_r = self.seq_length + self.burn_in - (k + pad_l)

        episode_dir = os.path.join(self.screen_dir, str(samples[self.burn_in]['ep']))
        s = self.preprocess_trajectory(episode_dir, samples[self.burn_in]['fr'] - l, k)

        r = np.stack([sample['r'] for sample in samples], axis=0)
        rho = np.stack([sample['rho'] for sample in samples], axis=0)
        a = np.stack([sample['a'] for sample in samples], axis=0)
        pi = np.stack([sample['pi'] for sample in samples], axis=0)
        h_adv = np.stack([sample['h_adv'] for sample in samples], axis=0)
        h_v = np.stack([sample['h_v'] for sample in samples], axis=0)
        h_beta = np.stack([sample['h_beta'] for sample in samples], axis=0)

        r = np.pad(r, [(pad_l, pad_r)], 'constant', constant_values=0)
        rho = np.pad(rho, [(pad_l, pad_r), (0, 0)], 'constant', constant_values=0)
        a = np.pad(a, [(pad_l, pad_r)], 'constant', constant_values=0)
        pi = np.pad(pi, [(pad_l, pad_r), (0, 0)], 'constant', constant_values=0)
        h_adv = np.pad(h_adv, [(pad_l, pad_r), (0, 0)], 'constant', constant_values=0)
        h_v = np.pad(h_v, [(pad_l, pad_r), (0, 0)], 'constant', constant_values=0)
        h_beta = np.pad(h_beta, [(pad_l, pad_r), (0, 0)], 'constant', constant_values=0)
        s = np.pad(s, [(pad_l, pad_r), (0, 0), (0, 0), (0, 0)], 'constant', constant_values=0)

        return {'s': s, 'r': r, 'rho': rho, 'a': a, 'pi': pi, 'h_adv': h_adv[0], 'h_v': h_v[0], 'h_beta': h_beta[0]}


class ObservationsRNNBatchSampler(object):

    def __init__(self, replay_dir):

        self.batch = args.batch

        self.screen_dir = os.path.join(replay_dir, "explore", "screen")
        self.trajectory_dir = os.path.join(replay_dir, "explore", "trajectory")
        self.list_old_path = os.path.join(replay_dir, "list", "old_explore")

        self.replay_updates_interval = args.replay_updates_interval
        self.replay_memory_size = args.replay_memory_size + args.replay_explore_size
        self.readlock = os.path.join(replay_dir, "list", "readlock_explore.npy")

    def __iter__(self):

        traj_old = 0
        replay_buffer = np.array([])
        sequence = np.arange(-args.burn_in, args.seq_length)

        while True:

            # load new memory

            fread = lock_file(self.readlock)
            traj_sorted = np.load(fread)
            fread.seek(0)
            np.save(fread, [])
            release_file(fread)

            if not len(traj_sorted):
                continue

            replay = np.concatenate([np.load(os.path.join(self.trajectory_dir, "%d.npy" % traj)) for traj in traj_sorted], axis=0)

            replay_buffer = np.concatenate([replay_buffer, replay], axis=0)[-self.replay_memory_size:]

            # save previous traj_old to file
            np.save(self.list_old_path, np.array([traj_old]))
            traj_old = replay_buffer[0]['traj']
            print("Old trajectory: %d" % traj_old)
            print("New Sample size is: %d" % len(replay))

            len_replay_buffer = len(replay_buffer)

            minibatches = min(self.replay_updates_interval, int(len_replay_buffer / self.batch))

            shuffle_indexes = np.random.choice(len_replay_buffer, minibatches * self.batch, replace=False)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):

                samples = np.clip(np.expand_dims(shuffle_indexes[i * self.batch:(i + 1) * self.batch], axis=1)
                                  + sequence, a_max=len_replay_buffer-1, a_min=0)

                yield replay_buffer[samples]

    def __len__(self):
        return np.inf