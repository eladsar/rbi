import atari_py
import numpy as np
import torch

from config import consts, args
import cv2

img_width = args.width
img_height = args.height
interpolation = cv2.INTER_LINEAR


class Env(object):

    def __init__(self):

        self.skip = args.skip

        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', np.random.randint(2**31))
        # self.ale.setInt('max_num_frames', consts.max_length[args.game])
        self.ale.setFloat('repeat_action_probability', 0)

        # self.ale.setInt('frame_skip', self.skip)
        # self.ale.setBool('color_averaging', True)

        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(consts.gym_game_dict[args.game]))
        self.actions, = np.nonzero(consts.actions[args.game])
        self.action_meanings = [consts.action_meanings[i] for i in self.actions]

        self.k = 0  # Internal step counter
        self.kk = 0 # within life step counter
        self.lives = 0  # Life counter (used in DeepMind training)
        self.score = 0
        self.frame = None
        self.s, self.r, self.t, self.ram = None, None, None, None
        self.aux = torch.zeros(1, 1)
        self.history_length = args.history_length
        self.buffer = [np.zeros((args.height, args.width), dtype=np.float32)] * self.history_length
        self.nop = consts.nop

        tail = int(np.log(0.01) / np.log(args.discount))

        if args.multiplay:
            self.max_length = consts.max_length[args.game] + tail * self.skip
        else:
            self.max_length = consts.max_length[args.game]

        self.image = None
        self.max_score = consts.max_score[args.game]

    def reset(self, ram=None):
        # Reset internal
        self.score = 0
        self.buffer = [np.zeros((args.height, args.width), dtype=np.float32)] * self.history_length
        # Process and return initial state
        self.ale.setInt('random_seed', np.random.randint(2 ** 31))
        self.ale.reset_game()
        self.lives = self.ale.lives()

        # random number of no-ops 0-28 + 4
        noop_n = 4 * np.random.randint(10)
        for i in range(noop_n):
            self.ale.act(self.nop)

        self.step(self.nop)
        self.k = 0
        self.kk = 0
        self.t = 0

        if ram is not None:
            self.ale.restoreState(ram)
        self.ram = self.ale.cloneState()

    def step(self, a):

        # Process state
        self.r = 0
        a = self.actions[a]
        o = []

        self.r += self.ale.act(a)
        self.r += self.ale.act(a)
        self.r += self.ale.act(a)
        o.append(self.ale.getScreenGrayscale())
        self.r += self.ale.act(a)
        o.append(self.ale.getScreenGrayscale())
        t = self.ale.game_over()

        # self.r = self.ale.act(a)
        # # get gray scale image
        # self.image = self.ale.getScreenGrayscale()
        # t = self.ale.game_over()

        # self.last_action = action
        self.t = int(t or (self.k * self.skip >= self.max_length) or (self.score >= self.max_score))

        self.image = np.maximum(o[0], o[1])
        self.frame = cv2.resize(self.image.astype(np.float32), (img_width, img_height), interpolation=interpolation) / 256.

        self.k += 1

        # put in buffer
        self.buffer.pop()
        self.buffer.insert(0, self.frame)

        if self.lives > self.ale.lives():
            self.kk = 0
        else:
            self.kk += 1

        self.lives = self.ale.lives()

        # make a tensor from stacked images
        state = np.stack(self.buffer, axis=0)
        state = torch.FloatTensor(state)
        self.s = state.unsqueeze(0)

        # episode first 6 sec with off signal
        self.aux = (1 if float(self.kk) > 90 else 0) * torch.ones(1, 1)

        self.score += self.r
        self.ram = self.ale.cloneState()

