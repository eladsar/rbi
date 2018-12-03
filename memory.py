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

        return {'s': s, 'r': sample['r'], 'rho': sample['rho'], 'a': sample['a'], 'pi': sample['pi'],
                'aux': sample['aux']}


class ObservationsBatchSampler(object):

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
                yield replay_buffer[shuffle_indexes[i * self.batch:(i + 1) * self.batch]]

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

        return {'s': s, 'r': sample['r'], 'a': sample['a'], 't': sample['t'],
                'aux': sample['aux'], 's_tag': s_tag, 'aux_tag': sample['aux']}


# change behavior in the test dataset case: no prioritized replay, n_steps should follow the train dataset
# class ObservationsBatchSampler(object):
#
#     def __init__(self, replay_dir):
#         self.batch = [args.batch_exploit, args.batch_explore]
#
#         self.screen_dir = [os.path.join(replay_dir, "exploit", "screen"), os.path.join(replay_dir, "explore", "screen")]
#         self.trajectory_dir = [os.path.join(replay_dir, "exploit", "trajectory"), os.path.join(replay_dir, "explore", "trajectory")]
#         self.list_old_path = [os.path.join(replay_dir, "list", "old_exploit"), os.path.join(replay_dir, "list", "old_explore")]
#
#         self.replay_updates_interval = args.replay_updates_interval
#         self.replay_memory_size = [args.replay_memory_size, args.replay_explore_size]
#         self.readlock = [os.path.join(replay_dir, "list", "readlock_exploit.npy"), os.path.join(replay_dir, "list", "readlock_explore.npy")]
#
#         self.sample_type = [[0] * args.batch_exploit, [1] * args.batch_explore]
#
#     def __iter__(self):
#
#         traj_old = [0, 0]
#         replay_buffer = [np.array([]), np.array([])]
#
#         while True:
#
#             for source in [0, 1]:
#
#                 # load new memory
#
#                 fread = lock_file(self.readlock[source])
#                 traj_sorted = np.load(fread)
#                 fread.seek(0)
#                 np.save(fread, [])
#                 release_file(fread)
#
#                 if not len(traj_sorted):
#                     continue
#
#                 replay = np.concatenate([np.load(os.path.join(self.trajectory_dir[source], "%d.npy" % traj)) for traj in traj_sorted], axis=0)
#
#                 replay_buffer[source] = np.concatenate([replay_buffer[source], replay], axis=0)[-self.replay_memory_size[source]:]
#
#                 # save previous traj_old to file
#                 np.save(self.list_old_path[source], np.array([traj_old[source]]))
#                 traj_old[source] = replay_buffer[source][0]['traj']
#                 print("Old trajectory: %d" % traj_old[source])
#                 print("New Sample size is: %d" % len(replay))
#
#             len_replay_buffer = [len(replay_buffer[source]) for source in [0, 1]]
#
#             minibatches = [min(self.replay_updates_interval, int(len_replay_buffer[source] / self.batch[source])) for source in [0, 1]]
#             minibatches = min(minibatches)
#             # shuffle_indexes = [len_replay_buffer[source] - 1 - np.random.permutation(minibatches * self.batch[source]) for source in [0, 1]]
#             shuffle_indexes = [np.random.choice(len_replay_buffer[source], minibatches*self.batch[source], replace=False) for source in [0, 1]]
#
#             print("Exploiter:Replay Buffer size is: %d" % len_replay_buffer[0])
#             print("Explorer:Replay Buffer size is: %d" % len_replay_buffer[1])
#
#             for i in range(minibatches):
#
#                 exploiter = replay_buffer[0][shuffle_indexes[0][i*self.batch[0]:(i + 1) * self.batch[0]]]
#                 exploiter = list(zip(exploiter, self.sample_type[0]))
#
#                 explorer = replay_buffer[1][shuffle_indexes[1][i * self.batch[1]:(i + 1) * self.batch[1]]]
#                 explorer = list(zip(explorer, self.sample_type[1]))
#
#                 yield list(itertools.chain(exploiter, explorer))
#
#     def __len__(self):
#         return np.inf


class DQNBatchSampler(object):

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
                [np.load(os.path.join(self.trajectory_dir, "%d.npy" % traj)) for traj in traj_sorted],
                axis=0)

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
                yield replay_buffer[shuffle_indexes[i * self.batch:(i + 1) * self.batch]]

    def __len__(self):
        return np.inf


# def my_collate(batch):
#
#     sample = {
#         "s": torch.FloatTensor([x["s"] for x in batch]),
#         "a": torch.LongTensor([x["a"] for x in batch]),
#         "r": torch.FloatTensor([x["r"] for x in batch]),
#         "rho": torch.FloatTensor([x["rho"] for x in batch]),
#         "pi": torch.FloatTensor([x["pi"] for x in batch]),
#         "aux": torch.FloatTensor([x["aux"] for x in batch])}
#
#     return sample
