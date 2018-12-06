import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args
from preprocess import get_mc_value, hinv_np, lock_file, release_file
import itertools
import cv2


img_width = args.width
img_height = args.height
interpolation = cv2.INTER_LINEAR  # cv2.INTER_AREA  #
imread_grayscale = cv2.IMREAD_GRAYSCALE


class MemoryRNN(torch.utils.data.Dataset):

    def __init__(self):
        super(MemoryRNN, self).__init__()
        self.history_length = args.history_length
        self.n_steps = args.n_steps
        self.total_seq = args.seq_length + args.burn_in
        self.history_mat = np.expand_dims(np.arange(self.total_seq), axis=1) + np.arange(self.history_length)

    def __len__(self):
        return args.n_tot

    def __getitem__(self, index):
        raise NotImplementedError

    def preprocess_trajectory(self, episode_dir, frame):

        frames = [os.path.join(episode_dir, "%d.png" % (frame + i))
                  for i in range(-self.history_length + 1, self.total_seq + 1)]

        imgs = np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32),
                                     (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frames], axis=0)
        return imgs[self.history_mat, :, :]


class ObservationsRNNMemory(MemoryRNN):

    def __init__(self, replay_dir):

        super(ObservationsRNNMemory, self).__init__()
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    def __getitem__(self, samples):

        episode_dir = os.path.join(self.screen_dir, str(samples[0]['ep']))
        s = self.preprocess_trajectory(episode_dir, samples[0]['fr'])

        r = np.stack([samples[i]['r'] for i in samples], axis=0)
        rho = np.stack([samples[i]['rho'] for i in samples], axis=0)
        a = np.stack([samples[i]['a'] for i in samples], axis=0)
        pi = np.stack([samples[i]['pi'] for i in samples], axis=0)
        h_adv = np.stack([samples[i]['h_adv'] for i in samples], axis=0)
        h_v = np.stack([samples[i]['h_v'] for i in samples], axis=0)
        h_beta = np.stack([samples[i]['h_beta'] for i in samples], axis=0)

        return {'s': s, 'r': r, 'rho': rho, 'a': a, 'pi': pi, 'h_adv': h_adv, 'h_v': h_v, 'h_beta': h_beta}


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
        sequence = np.arange(args.burn_in + args.seq_length)

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
                yield replay_buffer[shuffle_indexes[i * self.batch:(i + 1) * self.batch] + sequence]

    def __len__(self):
        return np.inf