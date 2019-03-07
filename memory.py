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

    def __init__(self, replay_dir):
        super(Memory, self).__init__()
        self.history_length = args.history_length
        self.n_steps = args.n_steps
        self.screen_dir = os.path.join(replay_dir, "explore", "screen")

    def __len__(self):
        return args.n_tot

    def __getitem__(self, sample):

        sample, next_sample = sample

        episode_dir = os.path.join(self.screen_dir, str(sample['ep']))
        s = self.preprocess_history(episode_dir, sample['fr'])

        if not sample['t']:
            s_tag = self.preprocess_history(episode_dir, next_sample['fr'])
        else:
            s_tag = np.zeros((4, 84, 84), dtype=np.float32)

        return {'s': torch.from_numpy(s), 'r': torch.from_numpy(np.array(sample['r'])),
                'a': torch.from_numpy(np.array(sample['a'])), 't': torch.from_numpy(np.array(sample['t'])),
                'pi': torch.from_numpy(sample['pi']),
                's_tag': torch.from_numpy(s_tag), 'pi_tag': torch.from_numpy(next_sample['pi']),
                'tde': torch.from_numpy(np.array(sample['tde']))}

    def preprocess_history(self, episode_dir, frame):

        frame0 = [os.path.join(episode_dir, "%d.png" % (frame - i)) for i in range(self.history_length)]
        return np.stack([(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32), (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frame0], axis=0)
        # o = [(cv2.resize(cv2.imread(f0, imread_grayscale).astype(np.float32), (img_width, img_height), interpolation=interpolation) / 256.) for f0 in frame0]
        # o.insert(0, np.zeros((img_width, img_height), dtype=np.float32))
        # o = np.stack(o, axis=0)
        # o = np.diff(o, axis=0)
        # o = o * img_width * img_height / o.sum(axis=(1, 2), keepdims=True) / 2
        # print(o.shape)
        # return o



class ReplayBatchSampler(object):

    def __init__(self, replay_dir):

        self.batch = args.batch

        self.screen_dir = os.path.join(replay_dir, "explore", "screen")
        self.trajectory_dir = os.path.join(replay_dir, "explore", "trajectory")
        self.list_old_path = os.path.join(replay_dir, "list", "old_explore")

        self.replay_updates_interval = args.replay_updates_interval
        self.replay_memory_size = args.replay_memory_size
        self.readlock = os.path.join(replay_dir, "list", "readlock_explore.npy")

        self.rec_type = consts.rec_type
        self.n_steps = args.n_steps

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

            minibatches = min(self.replay_updates_interval, int(len_replay_buffer / self.batch) - self.n_steps)

            tde = replay_buffer['tde'] / np.sum(replay_buffer['tde'])
            tde = tde[:-self.n_steps]

            shuffle_indexes = np.random.choice(len_replay_buffer - self.n_steps, (minibatches, self.batch),
                                               replace=True, p=tde)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):
                samples = shuffle_indexes[i]
                yield zip(replay_buffer[samples], replay_buffer[samples + self.n_steps])

    def __len__(self):
        return np.inf


def collate(batch):

    numel = sum([x['s'].numel() for x in batch])
    storage = batch[0]['s'].storage()._new_shared(numel)
    out_s = batch[0]['s'].new(storage)

    numel = sum([x['s_tag'].numel() for x in batch])
    storage = batch[0]['s_tag'].storage()._new_shared(numel)
    out_s_tag = batch[0]['s_tag'].new(storage)

    return {'s': torch.stack([sample['s'] for sample in batch], out=out_s),
            's_tag': torch.stack([sample['s_tag'] for sample in batch], out=out_s_tag),
            'a': torch.stack([sample['a'] for sample in batch]),
            'r': torch.stack([sample['r'] for sample in batch]),
            't': torch.stack([sample['t'] for sample in batch]),
            'pi': torch.stack([sample['pi'] for sample in batch]),
            'pi_tag': torch.stack([sample['pi_tag'] for sample in batch]),
            'tde': torch.stack([sample['tde'] for sample in batch])}