import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args
from preprocess import get_mc_value, hinv_np, lock_file, release_file
import itertools
import cv2
import pandas as pd
from logger import logger

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

        frames = [os.path.join(episode_dir, "%d.png" % max(frame + i, -1))
                  for i in range(-self.history_length + 1, k)]
        imgs = np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32),
                                         (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frames], axis=0)
        return imgs[self.history_mat[:k], :, :]


class ObservationsRNNMemory(MemoryRNN):

    def __init__(self, replay_dir):

        super(ObservationsRNNMemory, self).__init__()
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    # def __getitem__(self, samples):
    #
    #     # if the first episode is too short s.t. there is no hidden state sample, take the second episode
    #     samples.reset_index(drop=True, inplace=True)
    #
    #     if (samples['fr_e'][0] - samples['fr'][0]) <= (self.burn_in + (-samples['fr'][0] % self.seq_overlap)):
    #
    #         # take the second episode
    #         offset = samples['fr_e'][0] - samples['fr'][0]
    #         start = offset + (-samples['fr'][offset] % self.seq_overlap)
    #         samples = samples[start:start + self.burn_in + self.seq_length]
    #
    #     else:
    #         # take the first episode
    #         start = (-samples['fr'][0] % self.seq_overlap)
    #         end = min(samples['fr_e'][0] - samples['fr'][0], (self.burn_in + self.seq_length) + start)
    #         samples = samples[start:end]
    #
    #     samples.reset_index(drop=True, inplace=True)
    #
    #     prune_length = len(samples)
    #     pad_l = min(self.seq_length + self.burn_in - prune_length, self.burn_in)
    #     pad_r = self.seq_length + self.burn_in - (prune_length + pad_l)
    #
    #     episode_dir = os.path.join(self.screen_dir, str(samples['ep'][0]))
    #     s = self.preprocess_trajectory(episode_dir, samples['fr'][0], prune_length)
    #
    #     r = samples['r'].values.astype(np.float32)
    #     rho_v = np.stack(samples['rho_v'].values)
    #     rho_q = np.stack(samples['rho_q'].values)
    #     a = samples['a'].values
    #     pi = np.stack(samples['pi'].values)
    #
    #     h_q = samples['h_q'].values[0] if not pad_l else np.zeros(self.hidden_features, dtype=np.float32)
    #     h_beta = samples['h_beta'].values[0] if not pad_l else np.zeros(self.hidden_features, dtype=np.float32)
    #
    #     if h_q is None or h_beta is None:
    #         logger.info("Log exception")
    #         logger.info("episode: %d" % samples['ep'][0])
    #         logger.info("frame %d" % samples['fr'][0])
    #         logger.info("length %d" % prune_length)
    #         logger.info("fr_e %d" % samples['fr_e'][0])
    #
    #
    #     r = np.pad(r, [(pad_l, pad_r)], 'constant', constant_values=0)
    #     rho_v = np.pad(rho_v, [(pad_l, pad_r)], 'constant', constant_values=0)
    #     rho_q = np.pad(rho_q, [(pad_l, pad_r)], 'constant', constant_values=0)
    #     a = np.pad(a, [(pad_l, pad_r)], 'constant', constant_values=0)
    #     pi = np.pad(pi, [(pad_l, pad_r), (0, 0)], 'constant', constant_values=0)
    #     s = np.pad(s, [(pad_l, pad_r), (0, 0), (0, 0), (0, 0)], 'constant', constant_values=0)
    #
    #     return {'s': s[self.burn_in:], 'r': r[self.burn_in:], 'rho_q': rho_q[self.burn_in:], 'rho_v': rho_v[self.burn_in:],
    #             'a': a[self.burn_in:], 'pi': pi[self.burn_in:], 'h_q': h_q, 'h_beta': h_beta,
    #             's_bi': s[:self.burn_in],
    #             'a_bi': a[:self.burn_in]}

    def __getitem__(self, samples):

        # if the first episode is too short s.t. there is no hidden state sample, take the second episode

        # samples.reset_index(drop=True, inplace=True)
        #
        # episode_dir = os.path.join(self.screen_dir, str(samples['ep'][0]))
        # s = self.preprocess_history(episode_dir, samples['fr'][0])
        # s = np.expand_dims(s, axis=0)
        #
        # r = samples['r'].values.astype(np.float32)[0:1]
        # rho_v = np.stack(samples['rho_v'].values)[0:1]
        # rho_q = np.stack(samples['rho_q'].values)[0:1]
        # a = samples['a'].values[0:1]
        # pi = np.stack(samples['pi'].values)[0:1]
        #
        # h_q = np.zeros(self.hidden_features, dtype=np.float32)
        # h_beta = np.zeros(self.hidden_features, dtype=np.float32)

        h_q = np.zeros(self.hidden_features, dtype=np.float32)
        h_beta = np.zeros(self.hidden_features, dtype=np.float32)

        s = np.zeros((2,4,84,84), dtype=np.float32)
        a = np.zeros(2, dtype=np.int)
        r = np.zeros(2, dtype=np.float32)
        rho_q = np.ones(2, dtype=np.float32)
        rho_v = np.ones(2, dtype=np.float32)
        pi = np.zeros((2, 5), dtype=np.float32)


        return {'s': s, 'r': r, 'rho_q': rho_q, 'rho_v': rho_v,
                'a': a, 'pi': pi, 'h_q': h_q, 'h_beta': h_beta,}


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
        # replay_buffer = np.array([])
        replay_buffer = pd.DataFrame({})
        total_seq_length = args.burn_in + args.seq_length + args.seq_overlap
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

            # replay = np.concatenate([np.load(os.path.join(self.trajectory_dir, "%d.npy" % traj)) for traj in traj_sorted], axis=0)
            # replay_buffer = np.concatenate([replay_buffer, replay], axis=0)[-self.replay_memory_size:]

            replay = pd.concat(
                [pd.read_pickle(os.path.join(self.trajectory_dir, "%d.pkl" % traj)) for traj in traj_sorted], axis=0)

            replay_buffer = pd.concat([replay_buffer, replay], axis=0)
            replay_buffer.reset_index(drop=True, inplace=True)
            if len(replay_buffer) >= self.replay_memory_size:
                offset = replay_buffer.iloc[-self.replay_memory_size]['fr'] - replay_buffer.iloc[-self.replay_memory_size]['fr_s']
            else:
                offset = 0
            replay_buffer = replay_buffer.iloc[-self.replay_memory_size - offset:]
            replay_buffer.reset_index(drop=True, inplace=True)

            assert (replay_buffer.iloc[0]['fr'] == replay_buffer.iloc[0]['fr_s']), \
                "Missmatch start offset %d, %d, %d" % (offset, replay_buffer.iloc[0]['fr'], replay_buffer.iloc[0]['fr_s'])

            assert (replay_buffer.iloc[-1]['fr'] + 1 == replay_buffer.iloc[-1]['fr_e']), \
                "Missmatch end %d, %d" %  (replay_buffer.iloc[-1]['fr'], replay_buffer.iloc[-1]['fr_e'])

            # save previous traj_old to file
            np.save(self.list_old_path, np.array([traj_old]))
            traj_old = replay_buffer['traj'][0]
            print("Old trajectory: %d" % traj_old)
            print("New Sample size is: %d" % len(replay))

            len_replay_buffer = len(replay_buffer) - total_seq_length

            minibatches = min(self.replay_updates_interval, int(len_replay_buffer / self.batch))

            shuffle_indexes = np.random.choice(len_replay_buffer, minibatches * self.batch, replace=False)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):

                samples = np.expand_dims(shuffle_indexes[i * self.batch:(i + 1) * self.batch], axis=1) + sequence
                cond = np.logical_and(samples >= 0, samples < len_replay_buffer)

                # yield replay_buffer[samples]
                # yield [replay_buffer.iloc[samples[i][cond[i]]] for i in range(self.batch)]
                # yield [replay_buffer.iloc[samples[i]].copy(deep=True) for i in range(self.batch)]
                # yield [1 for i in range(self.batch)]
                # yield replay_buffer.iloc[range(self.batch)]
                # yield np.arange(self.batch)

                yield map(lambda x: replay_buffer.iloc[samples[x][cond[x]]], range(self.batch))

    def __len__(self):
        return np.inf