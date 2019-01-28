import numpy as np
import os
from config import consts, args
import torch

class Agent(object):

    def __init__(self, root_dir, checkpoint=None):
        self.model = None
        self.optimizer = None
        # parameters
        self.discount = args.discount
        self.update_target_interval = args.update_target_interval
        self.update_memory_interval = args.update_memory_interval
        self.load_memory_interval = args.load_memory_interval
        self.action_space = len(np.nonzero(consts.actions[args.game])[0])
        self.skip = args.skip
        self.termination_reward = args.termination_reward
        self.n_steps = args.n_steps
        self.reward_shape = args.reward_shape
        self.player_replay_size = args.player_replay_size
        self.cmin = args.cmin
        self.cmax = args.cmax
        self.history_length= args.history_length
        self.random_initialization = args.random_initialization
        self.epsilon = args.epsilon * self.action_space / (self.action_space - 1)
        self.delta = args.delta
        self.player = args.player
        self.priority_beta = args.priority_beta
        self.priority_alpha = args.priority_alpha
        self.epsilon_a = args.epsilon_a
        self.cuda_id = args.cuda_default
        self.behavioral_avg_frame = 1
        self.behavioral_avg_score = -1
        self.entropy_loss = float((1 - (1 / (1 + (self.action_space - 1) * np.exp(-args.softmax_diff)))) * (self.action_space / (self.action_space - 1)))
        self.batch = args.batch
        self.replay_memory_size = args.replay_memory_size
        self.n_actors = args.n_actors
        self.actor_index = args.actor_index
        self.n_players = args.n_players
        self.player = args.player
        self.n_tot = args.n_tot
        self.max_length = consts.max_length[args.game]
        self.max_score = consts.max_score[args.game]
        self.start_time = consts.start_time

        self.mix = self.delta
        self.min_loop = 1. / 44
        self.hidden_state = args.hidden_features_rnn

        self.seq_length = args.seq_length
        if args.target == 'tde':
            self.seq_length += self.n_steps

        self.burn_in = args.burn_in
        self.seq_overlap = args.seq_overlap

        self.rec_type = consts.rec_type

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

    def save_checkpoint(self, path, aux=None):
        raise NotImplementedError

    def load_checkpoint(self, path):

        raise NotImplementedError

    def train(self, n_interval, n_tot):
        raise NotImplementedError

    def evaluate(self, n_interval, n_tot):
        raise NotImplementedError

    def set_player(self, player, cmin=None, cmax=None, delta=None,
                   epsilon=None, behavioral_avg_score=None,
                   behavioral_avg_frame=None, explore_threshold=None):

        self.player = player

        if epsilon is not None:
            self.epsilon = epsilon * self.action_space / (self.action_space - 1)

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

    def resume(self, model_path):
        aux = self.load_checkpoint(model_path)
        return aux
