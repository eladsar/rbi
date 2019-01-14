import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args
from preprocess import lock_file, release_file
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
        self.reward_length = args.seq_length

        if args.target == 'tde':
            self.seq_length += self.n_steps
        else:
            self.n_steps = 0

        self.burn_in = args.burn_in
        self.history_mat = np.expand_dims(np.arange(self.seq_length + self.burn_in), axis=1) + np.arange(self.history_length)
        self.history_mat = np.fliplr(self.history_mat)

        self.hidden_features = args.hidden_features_rnn
        self.seq_overlap = args.seq_overlap

    def __len__(self):
        return args.n_tot

    def __getitem__(self, index):
        raise NotImplementedError

    def preprocess_history(self, episode_dir, frame):

        frame0 = [os.path.join(episode_dir, "%d.png" % (frame - i)) for i in range(self.history_length)]
        return np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32), (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frame0], axis=0)

    def preprocess_trajectory(self, episode_dir, frame, k):

        frames = [os.path.join(episode_dir, "%d.png" % (frame + i)) for i in range(-self.history_length + 1, k)]
        imgs = np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32),
                                         (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frames], axis=0)
        return imgs[self.history_mat[:k], :, :]


class ObservationsRNNMemory(MemoryRNN):

    def __init__(self, replay_dir):

        super(ObservationsRNNMemory, self).__init__()
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    def __getitem__(self, samples):

        # if the first episode is too short s.t. there is no hidden state sample, take the second episode

        tde = samples['tde'][self.seq_length + self.burn_in]
        if (samples['fr_e'][0] - samples['fr'][0]) <= (self.burn_in + (-samples['fr'][0] % self.seq_overlap)):

            # take the second episode
            offset = samples['fr_e'][0] - samples['fr'][0]
            start = offset + (-samples['fr'][offset] % self.seq_overlap)
            samples = samples[start:start + self.burn_in + self.seq_length]

        else:
            # take the first episode
            start = (-samples['fr'][0] % self.seq_overlap)
            end = min(samples['fr_e'][0] - samples['fr'][0], (self.burn_in + self.seq_length) + start)
            samples = samples[start:end]

        prune_length = len(samples)
        pad_l = min(self.seq_length + self.burn_in - prune_length, self.burn_in)
        pad_r = self.seq_length + self.burn_in - (prune_length + pad_l)

        episode_dir = os.path.join(self.screen_dir, str(samples['ep'][0]))
        s = self.preprocess_trajectory(episode_dir, samples['fr'][0], prune_length)

        h_q = samples['h_q'][0]
        h_beta = samples['h_beta'][0]

        r = np.pad(samples['r'], [(0, pad_r + pad_l)], 'constant', constant_values=0)
        rho_v = np.pad(samples['rho_v'], [(0, pad_r + pad_l)], 'constant', constant_values=0)
        rho_q = np.pad(samples['rho_q'], [(0, pad_r + pad_l)], 'constant', constant_values=0)
        a = np.pad(samples['a'], [(0, pad_r + pad_l)], 'constant', constant_values=0)
        pi = np.pad(np.stack(samples['pi']), [(0, pad_r + pad_l), (0, 0)], 'constant', constant_values=0)
        s = np.pad(s, [(0, pad_r + pad_l), (0, 0), (0, 0), (0, 0)], 'constant', constant_values=0)
        t = np.pad(samples['t'], [(0, pad_r + pad_l)], 'constant', constant_values=1)


        return {'s': s[self.burn_in:], 'r': r[-self.seq_length:-self.n_steps], 'rho_q': rho_q[-self.seq_length:-self.n_steps],
                'rho_v': rho_v[self.burn_in:], 'a': a[self.burn_in:], 'pi': pi[self.burn_in:],
                'h_q': h_q, 'h_beta': h_beta, 's_bi': s[:self.burn_in], 't': t[self.burn_in:],
                'tde': tde}


class ObservationsRNNBatchSampler(object):

    def __init__(self, replay_dir):

        self.batch = args.batch

        self.screen_dir = os.path.join(replay_dir, "explore", "screen")
        self.trajectory_dir = os.path.join(replay_dir, "explore", "trajectory")
        self.list_old_path = os.path.join(replay_dir, "list", "old_explore")

        self.replay_updates_interval = args.replay_updates_interval
        self.replay_memory_size = args.replay_memory_size
        self.readlock = os.path.join(replay_dir, "list", "readlock_explore.npy")

        self.rec_type = consts.rec_type

        self.priority_alpha = args.priority_alpha
        self.epsilon_a = args.epsilon_a

        self.tde = np.array([])

    def __iter__(self):

        traj_old = 0
        replay_buffer = np.array([], dtype=self.rec_type)

        total_seq_length = args.burn_in + args.seq_length + args.seq_overlap

        if args.target == 'tde':
            total_seq_length += args.n_steps

        sequence = np.arange(total_seq_length)

        while True:

            # load new memory
            fread = lock_file(self.readlock)
            traj_sorted = np.load(fread)
            fread.seek(0)
            np.save(fread, [])
            release_file(fread)

            if not len(traj_sorted):
                continue

            replay = np.concatenate(
                [np.load(os.path.join(self.trajectory_dir, "%d.npy" % traj)) for traj in traj_sorted], axis=0)
            replay_buffer = np.concatenate([replay_buffer, replay], axis=0)

            offset = replay_buffer[-self.replay_memory_size]['fr'] - replay_buffer[-self.replay_memory_size]['fr_s'] \
                if len(replay_buffer) >= self.replay_memory_size else 0

            replay_buffer = replay_buffer[-self.replay_memory_size - offset:]

            tde = replay_buffer['tde'][total_seq_length:]
            tde = (tde + self.epsilon_a) ** self.priority_alpha
            self.tde = tde / tde.sum()

            # save previous traj_old to file
            np.save(self.list_old_path, np.array([traj_old]))
            traj_old = replay_buffer['traj'][0]
            print("Old trajectory: %d" % traj_old)
            print("New Sample size is: %d" % len(replay))

            len_replay_buffer = len(replay_buffer) - total_seq_length
            minibatches = min(self.replay_updates_interval, int(len_replay_buffer / self.batch))

            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False,
                                               p=self.tde)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):

                samples = shuffle_indexes[i, :, None] + sequence
                yield replay_buffer[samples]

    def __len__(self):
        return np.inf