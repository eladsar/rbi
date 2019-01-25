import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args
from preprocess import lock_file, release_file
import itertools
import cv2


img_width = args.width
img_height = args.height
interpolation = cv2.INTER_LINEAR  # cv2.INTER_AREA  #
imread_grayscale = cv2.IMREAD_GRAYSCALE


class Memory(torch.utils.data.Dataset):

    def __init__(self):
        super(Memory, self).__init__()
        self.history_length = args.history_length
        self.n_steps = args.n_steps

    def __len__(self):
        return args.n_tot

    def __getitem__(self, index):
        raise NotImplementedError

    def preprocess_history(self, episode_dir, frame):

        frame0 = [os.path.join(episode_dir, "%d.png" % (frame - i)) for i in range(self.history_length)]
        return np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32), (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frame0], axis=0)


class ObservationsMemory(Memory):

    def __init__(self, replay_dir):

        super(ObservationsMemory, self).__init__()
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    def __getitem__(self, sample):

        episode_dir = os.path.join(self.screen_dir, str(sample['ep']))
        s = self.preprocess_history(episode_dir, sample['fr'])

        return {'s': s, 'r': sample['r'], 'rho_v': sample['rho_v'],
                'rho_q': sample['rho_q'], 'a': sample['a'], 'pi': sample['pi'],
                'aux': sample['aux']}


class ObservationsBatchSampler(object):

    def __init__(self, replay_dir):

        self.batch = args.batch

        self.screen_dir = os.path.join(replay_dir, "explore", "screen")
        self.trajectory_dir = os.path.join(replay_dir, "explore", "trajectory")
        self.list_old_path = os.path.join(replay_dir, "list", "old_explore")

        self.replay_updates_interval = args.replay_updates_interval
        self.replay_memory_size = args.replay_memory_size
        self.readlock = os.path.join(replay_dir, "list", "readlock_explore.npy")

        self.rec_type = consts.rec_type

    def __iter__(self):

        traj_old = 0
        replay_buffer = np.array([], dtype=self.rec_type)

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
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):
                samples = shuffle_indexes[i]
                yield replay_buffer[samples]

    def __len__(self):
        return np.inf


class DQNMemory(Memory):

    def __init__(self, replay_dir):

        super(DQNMemory, self).__init__()
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    def __getitem__(self, sample):

        episode_dir = os.path.join(self.screen_dir, str(sample['ep']))
        s = self.preprocess_history(episode_dir, sample['fr'])

        if not sample['t']:
            s_tag = self.preprocess_history(episode_dir, sample['fr']+self.n_steps)
        else:
            s_tag = np.zeros((4, 84, 84), dtype=np.float32)

        return {'s': s, 'r': sample['r'], 'a': sample['a'], 't': sample['t'], 'pi': sample['pi'],
                'aux': sample['aux'], 's_tag': s_tag, 'aux_tag': sample['aux']}


DQNBatchSampler = ObservationsBatchSampler

