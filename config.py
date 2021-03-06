import argparse
import time
import numpy as np
import socket
import os
import pwd

parser = argparse.ArgumentParser(description='atari')
username = pwd.getpwuid(os.geteuid()).pw_name

if "gpu" in socket.gethostname():
    base_dir = os.path.join('/home/dsi/', username, 'data/rbi')
elif "root" == username:
    base_dir = r'/workspace/data/rbi/'
else:
    base_dir = os.path.join('/data/', username, 'rbi_atari')


def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})


# Arguments

# strings
parser.add_argument('--game', type=str, default='nogame', help='ATARI game')
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')
parser.add_argument('--algorithm', type=str, default='rbi_rnn', help='[rbi|ppo|ape|rbi_rnn]')
parser.add_argument('--base-dir', type=str, default=base_dir, help='Base directory for Logs and results')


# booleans
boolean_feature("load-last-model", False, 'Load the last saved model')
boolean_feature("load-best-model", False, 'Load the best saved model')
boolean_feature("learn", False, 'Learn from the observations')
boolean_feature("play", False, 'Test the learned model via playing')
boolean_feature("postprocess", False, 'Postprocess evaluation results')
boolean_feature("multiplay", False, 'Send samples to memory from multiple parallel players')
boolean_feature("evaluate", False, 'evaluate player')
boolean_feature("clean", False, 'Clean old trajectories')
boolean_feature("tensorboard", True, "Log results to tensorboard")
boolean_feature("log-scores", True, "Log score results to NPY objects")
parser.add_argument('--n-steps', type=int, default=6, metavar='STEPS', help='Number of steps for multi-step learning')

# my models parameters (Elad)
boolean_feature("dropout", False, "Use Dropout layer")
boolean_feature("reward-shape", False, "Shape reward with sign(r)*log(1+|r|)")
boolean_feature("infinite-horizon", False, "Don't end episode in EOL")
parser.add_argument('--target', type=str, default='tde', help='Target method [td|tde|mc]')

# parameters
parser.add_argument('--resume', type=int, default=-1, help='Resume experiment number, set -1 for last experiment')

# model parameters
parser.add_argument('--wait', type=float, default=0, help='Sleep at start-time')
parser.add_argument('--skip', type=int, default=4, help='Skip pattern')
parser.add_argument('--height', type=int, default=84, help='Image Height')
parser.add_argument('--width', type=int, default=84, help='Image width')
parser.add_argument('--batch', type=int, default=128, help='Mini-Batch Size')
parser.add_argument('--max-frame', type=int, default=50000, help='Episode Frame Limit')

parser.add_argument('--seq-length', type=int, default=30, help='Length of RNN sequence')
parser.add_argument('--burn-in', type=int, default=10, help='Length of Burn-in RNN sequence')
parser.add_argument('--seq-overlap', type=int, default=10, help='Overlap between two consecutive squences')

parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-features', type=int, default=512, metavar='N', help='Number of hidden features in (CNN output)')
parser.add_argument('--hidden-features-rnn', type=int, default=512, metavar='N', help='Number of hidden features in (RNN output)')
parser.add_argument('--play-episodes-interval', type=int, default=16, metavar='N', help='Number of episodes between net updates')

parser.add_argument('--clip', type=float, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--termination-reward', type=float, default=0, help='Reward for terminal state')
parser.add_argument('--friction-reward', type=float, default=0, help='Negative friction reward')

parser.add_argument('--priority-alpha', type=float, default=0.5, metavar='α', help='Attenuation factor for the prioritized replay distribution')
parser.add_argument('--priority-beta', type=float, default=0.5, metavar='β', help='Priority importance sampling coefficient')
parser.add_argument('--priority-eta', type=float, default=0.9, metavar='η', help='Priority time series coefficient')
parser.add_argument('--epsilon-a', type=float, default=0.001, metavar='ε', help='Priority replay epsilon-a')
parser.add_argument('--cmin', type=float, default=0.1, metavar='c_min', help='Lower reroute threshold')
parser.add_argument('--cmax', type=float, default=2, metavar='c_max', help='Upper reroute threshold')
parser.add_argument('--delta', type=float, default=0.1, metavar='delta', help='Total variation constraint')
parser.add_argument('--ppo-eps', type=float, default=0.1, metavar='ε', help='PPO epsilon level')

parser.add_argument('--player', type=str, default='reroutetv', help='Player type: [reroute/tv]')

# exploration parameters
parser.add_argument('--softmax-diff', type=float, default=3.8, metavar='β', help='Maximum softmax diff')
parser.add_argument('--epsilon', type=float, default=0.00164, metavar='ε', help='exploration parameter before behavioral period')

# dataloader
parser.add_argument('--cpu-workers', type=int, default=48, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda-default', type=int, default=0, help='Default GPU')

# train parameters
parser.add_argument('--update-target-interval', type=int, default=2500, metavar='STEPS', help='Number of traning iterations between q-target updates')
parser.add_argument('--n-tot', type=int, default=3125000, metavar='STEPS', help='Total number of training steps')
parser.add_argument('--checkpoint-interval', type=int, default=5000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--random-initialization', type=int, default=2500, metavar='STEPS', help='Number of training steps in random policy')
parser.add_argument('--player-replay-size', type=int, default=2500, help='Player\'s replay memory size')
parser.add_argument('--update-memory-interval', type=int, default=100, metavar='STEPS', help='Number of steps between memory updates')
parser.add_argument('--load-memory-interval', type=int, default=250, metavar='STEPS', help='Number of steps between memory loads')
parser.add_argument('--replay-updates-interval', type=int, default=5000, metavar='STEPS', help='Number of training iterations between q-target updates')
parser.add_argument('--replay-memory-size', type=int, default=2000000, help='Total replay exploit memory size')

# actors parameters
parser.add_argument('--n-players', type=int, default=16, help='Number of parallel players for current actor')
parser.add_argument('--actor-index', type=int, default=0, help='Index of current actor')
parser.add_argument('--n-actors', type=int, default=1, help='Total number of parallel actors')


# distributional learner

args = parser.parse_args()


# consts
class Consts(object):

    start_time = time.time()
    exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    api_key = "jthyXB1jO4czVy63ntyWZSnlf"

    action_space = 18
    nop = 0

    mem_threshold = int(5e9)

    action_meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
                       'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE',
                       'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE',
                       'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

    gym_game_dict = {"spaceinvaders": "space_invaders",
                     "mspacman": "ms_pacman",
                     "pinball": "video_pinball",
                     "qbert": "qbert",
                      "breakout": "breakout",
                     "seaquest": "seaquest",
                     "freeway": "freeway",
                     "enduro": "enduro",
                     "asterix": "asterix",
                     "berzerk": "berzerk",
                     "frostbite": "frostbite",
                     "icehockey": "ice_hockey",
                     "kangaroo": "kangaroo",
                     "revenge": "montezuma_revenge"}

    actions = {        "spaceinvaders": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       "revenge":       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "pinball":       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       "qbert":         [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       "mspacman":      [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       "seaquest":      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "enduro":        [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                       "asterix":       [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       "berzerk":       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "frostbite":     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "icehockey":     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "kangaroo":      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "breakout":      [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       "freeway":       [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    }

    max_length = {"spaceinvaders":      args.max_frame,
                       "revenge":       args.max_frame,
                       "pinball":       args.max_frame,
                       "qbert":         args.max_frame,
                       "seaquest":      args.max_frame,
                       "mspacman":      args.max_frame,
                       "freeway":       args.max_frame,
                       "breakout":      args.max_frame,
                       "enduro":        args.max_frame,
                       "asterix":       args.max_frame,
                       "berzerk":       args.max_frame,
                       "frostbite":     args.max_frame,
                       "icehockey":     args.max_frame,
                       "kangaroo":      args.max_frame,
    }

    max_score = {"spaceinvaders":       np.inf,
                       "revenge":       np.inf,
                       "pinball":       np.inf,
                       "qbert":         np.inf,
                       "seaquest":      np.inf,
                       "mspacman":      np.inf,
                       "freeway":       np.inf,
                       "breakout":      864,
                       "enduro":        np.inf,
                       "asterix":       np.inf,
                       "berzerk":       np.inf,
                       "frostbite":     np.inf,
                       "icehockey":     np.inf,
                       "kangaroo":      np.inf,
    }

    scale_reward = {"spaceinvaders":      5,
                       "revenge":       100,
                       "pinball":       100,
                       "qbert":         25,
                       "seaquest":      20,
                       "mspacman":      10,
                       "breakout":      1,
                        "enduro":       1,
                        "asterix":       50,
                        "berzerk":       50,
                        "freeway":       1,
                        "frostbite":     10,
                        "icehockey":     1,
                        "kangaroo":     100,
    }

    print("Game: %s" % args.game)

    rec_type = np.dtype([('fr', np.int), ('a', np.int), ('pi', np.float32, len(np.nonzero(actions[args.game])[0])),
                             ('h_beta', np.float32, args.hidden_features_rnn), ('h_q', np.float32, args.hidden_features_rnn),
                             ('ep', np.int), ('t', np.float32), ('fr_s', np.int), ('fr_e', np.int),
                             ('r', np.float32), ('rho_v', np.float32), ('rho_q', np.float32), ('traj', np.int),
                             ('tde', np.float32), ('aux', np.float32)])

    outdir = os.path.join(args.base_dir, 'results')
    logdir = os.path.join(args.base_dir, 'logs')
    indir = os.path.join('/dev/shm/', username, 'rbi')


consts = Consts()
