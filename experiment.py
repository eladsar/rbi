import csv
import time
import os
import sys
import numpy as np

from tensorboardX import SummaryWriter

from tqdm import tqdm
import time

from config import consts, args
from rbi_agent import RBIAgent
# from ape_encoded_agent import ApeAgent
from r2d2_agent import R2D2Agent
from ape_agent import ApeAgent
from ppo_agent import PPOAgent
from rbi_rnn_agent import RBIRNNAgent

from logger import logger
from distutils.dir_util import copy_tree


class Experiment(object):

    def __init__(self, logger_file):

        # parameters

        dirs = os.listdir(consts.outdir)

        self.load_model = args.load_last_model or args.load_best_model
        self.load_best = args.load_best_model
        self.load_last = args.load_last_model
        self.resume = args.resume
        self.action_meanings = [consts.action_meanings[i] for i in np.nonzero(consts.actions[args.game])[0]]
        self.log_scores = args.log_scores

        self.exp_name = ""
        if self.load_model:
            if self.resume >= 0:
                for d in dirs:
                    if "%s_exp_%04d_" % (args.identifier, self.resume) in d:
                        self.exp_name = d
                        self.exp_num = self.resume
                        break
            else:
                raise Exception("Non-existing experiment")

        if not self.exp_name:
            # count similar experiments
            n = sum([1 for d in dirs if "%s_exp" % args.identifier in d])
            self.exp_name = "%s_exp_%04d_%s" % (args.identifier, n, consts.exptime)
            self.load_model = False

            self.exp_num = n

        # init experiment parameters
        self.root = os.path.join(consts.outdir, self.exp_name)
        self.indir = consts.indir

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.scores_dir = os.path.join(self.root, 'scores')
        self.code_dir = os.path.join(self.root, 'code')
        self.analysis_dir = os.path.join(self.root, 'analysis')
        self.checkpoint = os.path.join(self.checkpoints_dir, 'checkpoint')
        self.checkpoint_best = os.path.join(self.checkpoints_dir, 'checkpoint_best')
        self.replay_dir = os.path.join(self.indir, self.exp_name)

        if self.load_model:
            logger.info("Resuming existing experiment")
            with open("logger", "a") as fo:
                fo.write("%s resume\n" % logger_file)
        else:
            logger.info("Creating new experiment")
            os.makedirs(self.root)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.results_dir)
            os.makedirs(self.scores_dir)
            os.makedirs(self.code_dir)
            os.makedirs(self.analysis_dir)
            # copy code to dir
            copy_tree(os.path.abspath("."), self.code_dir)

            # write csv file of hyper-parameters
            filename = os.path.join(self.root, "hyperparameters.csv")
            with open(filename, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(self.exp_name)
                for k, v in vars(args).items():
                    spamwriter.writerow([k, str(v)])

            with open(os.path.join(self.root, "logger"), "a") as fo:
                fo.write("%s\n" % logger_file)

        try:
            os.makedirs(self.replay_dir)

        except FileExistsError:
            pass

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()

    def choose_agent(self):

        if args.algorithm == "rbi":
            return RBIAgent
        elif args.algorithm == "rbi_rnn":
            return RBIRNNAgent
        elif args.algorithm == "ape":
            return ApeAgent
        elif args.algorithm == "ppo":
            return PPOAgent
        elif args.algorithm == "r2d2":
            return R2D2Agent
        else:
            return NotImplementedError

    def learn(self):

        # init time variables

        agent = self.choose_agent()(self.replay_dir, checkpoint=self.checkpoint)

        # load model
        if self.load_model:
            if self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError
            n_offset = aux['n']
        else:
            n_offset = 0
            # save a random init checkpoint
            agent.save_checkpoint(self.checkpoint, {'n': 0})

        # define experiment generators
        learn = agent.learn(args.checkpoint_interval, args.n_tot)
        agent.save_checkpoint(agent.snapshot_path, {'n': agent.n_offset})

        batch_explore = args.batch

        hold = 1
        while hold:
            print("wait for first samples")

            if len(os.listdir(os.path.join(self.replay_dir, "explore", "trajectory"))) >= (int(500. / args.player_replay_size * batch_explore) + 1):
                hold = 0

            time.sleep(5)

        logger.info("Begin Behavioral Distributional learning experiment")
        logger.info("Game: %s " % args.game)

        for n, train_results in enumerate(learn):

            n = n * args.checkpoint_interval

            avg_train_loss_beta = np.mean(train_results['loss_beta'])
            avg_train_loss_v_beta = np.mean(train_results['loss_value'])
            avg_train_loss_std = np.mean(train_results['loss_std'])

            avg_act_diff = np.mean(train_results['act_diff'])

            Hbeta = np.mean(train_results['Hbeta'])
            Hpi = np.mean(train_results['Hpi'])

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('train_loss/loss_beta', float(avg_train_loss_beta), n + n_offset)
                self.writer.add_scalar('train_loss/loss_value', float(avg_train_loss_v_beta), n + n_offset)
                self.writer.add_scalar('train_loss/loss_std', float(avg_train_loss_std), n + n_offset)

                self.writer.add_image('states/state', train_results['image'], n)
                self.writer.add_scalar('actions/act_diff', float(avg_act_diff), n + n_offset)

                self.writer.add_histogram("actions/agent", train_results['a_agent'], n + n_offset, 'doane')
                self.writer.add_histogram("actions/a_player", train_results['a_player'], n + n_offset, 'doane')

                if hasattr(agent, "beta_net"):
                    for name, param in agent.beta_net.named_parameters():
                        self.writer.add_histogram("beta_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("value_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')

                if hasattr(agent, "dqn_net"):
                    for name, param in agent.dqn_net.named_parameters():
                        self.writer.add_histogram("value_net/%s" % name, param.clone().cpu().data.numpy(), n + n_offset,
                                                  'fd')

            # img = train_results['s'][0, :-1, :, :]
            # self.writer.add_image('states/state', img, n + n_offset)
            self.print_actions_statistics(train_results['a_agent'], train_results['a_player'], n + n_offset, Hbeta, Hpi,
                                          train_results['adv_a'], train_results['q_a'], train_results['mc_val'])

            player = self.get_player(agent)
            score = player['high'] if player else 0
            agent.save_checkpoint(self.checkpoint, {'n': n + n_offset, 'score': score})

        return agent

    def get_player(self, agent):

        if os.path.isdir(agent.best_player_dir) and os.listdir(agent.best_player_dir):
            max_n = 0

            for stat_file in os.listdir(agent.best_player_dir):

                while True:
                    try:
                        data = np.load(os.path.join(agent.best_player_dir, stat_file)).item()
                        break
                    except OSError:
                        time.sleep(0.1)

                if max_n <= data['n']:
                    max_n = data['n']
                    player_stats = data['statistics']

            # fix choice
            for pt in player_stats:
                if pt == "reroute":
                    player_type = pt
                    break

            return player_stats[player_type]

        return None

    def multiplay(self):

        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)
        multiplayer = agent.multiplay()

        while True:

            player = self.get_player(agent)
            if player:

                agent.set_player(player['player'], behavioral_avg_score=player['high'], behavioral_avg_frame=player['frames'])

            next(multiplayer)

    def play(self, params=None):

        uuid = "%012d" % np.random.randint(1e12)
        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)
        aux = agent.resume(self.checkpoint)

        n = aux['n']
        results = {"n" : n, "score": [], "frame": []}

        player = agent.play(args.play_episodes_interval, save=False, load=False, fix=True)

        for i, step in tqdm(enumerate(player)):
            results["score"].append(step['score'])
            results["frame"].append(step['frames'])
            print("frames: %d | score: %d |" % (step['frames'], step['score']))

        if self.log_scores:
            logger.info("Save NPY file: eval_%d_%s.npy" % (n, uuid))
            filename = os.path.join(self.scores_dir, "eval_%d_%s" % (n, uuid))
            np.save(filename, results)

    def clean(self):

        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint, choose=True)
        agent.clean()

    def choose(self):

        uuid = "%012d" % np.random.randint(1e12)
        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint, choose=True)

        tensorboard_path = os.path.join(self.results_dir, uuid)
        os.makedirs(tensorboard_path)

        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=tensorboard_path, comment="%s_%s" % (args.identifier, uuid))

        results_filename = os.path.join(agent.best_player_dir, "%s.npy" % uuid)
        scores_dir = os.path.join(self.scores_dir, uuid)
        os.makedirs(scores_dir)

        kk = 0

        if args.algorithm in ["rbi", "rbi_rnn"]:
            results = {'n': 0,
                       'statistics': {
                           'reroute': {'player': 'reroutetv', 'cmin': args.cmin, 'cmax': args.cmax, 'delta': args.delta, 'score': 0, 'high': 0, 'frames':1},
                           'behavioral': {'player': 'behavioral', 'cmin': None, 'cmax': None, 'delta': 0, 'score': 0, 'high': 0, 'frames':1}
                       }}
        elif args.algorithm in ["ape", "r2d2"]:
            results = {'n': 0,
                       'statistics': {
                           'reroute': {'player': 'reroutetv', 'cmin': args.cmin, 'cmax': args.cmax, 'delta': args.delta, 'score': 0, 'high': 0, 'frames':1},
                           'behavioral': {'player': 'behavioral', 'cmin': None, 'cmax': None, 'delta': 0, 'score': 0, 'high': 0, 'frames':1}
                       }}
        elif args.algorithm == "ppo":
            results = {'n': 0,
                       'statistics': {
                           'reroute': {'player': 'reroutetv', 'cmin': args.cmin, 'cmax': args.cmax, 'delta': args.delta, 'score': 0, 'high': 0,
                                                  'frames': 1},
                           'behavioral': {'player': 'behavioral', 'cmin': None, 'cmax': None, 'delta': 0, 'score': 0, 'high': 0,
                                          'frames': 1}
                       }}
        else:
            raise NotImplementedError

        time.sleep(args.wait)

        print("Here")

        while True:

            # load model
            try:
                aux = agent.resume(agent.snapshot_path)
            except:  # when reading and writing collide
                time.sleep(2)
                aux = agent.resume(agent.snapshot_path)

            n = aux['n']

            if n < args.random_initialization:
                time.sleep(5)
                continue

            results['n'] = n
            results['time'] = time.time() - consts.start_time

            for player_name in results['statistics']:

                scores = []
                frames = []
                mc = np.array([])
                q = np.array([])

                player_params = results['statistics'][player_name]
                agent.set_player(player_params['player'], cmin=player_params['cmin'], cmax=player_params['cmax'],
                                 delta=player_params['delta'])

                player = agent.play(args.play_episodes_interval, save=False, load=False)

                stats = {"score": [], "frame": [], "time": [], "n": []}

                tic = time.time()

                for i, step in enumerate(player):

                    print("stats | player: %s | episode: %d | time: %g" % (player_name, i, time.time() - tic))
                    tic = time.time()
                    scores.append(step['score'])
                    frames.append(step['frames'])
                    mc = np.concatenate((mc, step['mc']))
                    q = np.concatenate((q, step['q']))

                    # add stats results
                    stats["score"].append(step['score'])
                    stats["frame"].append(step['frames'])
                    stats["n"].append(step['n'])
                    stats["time"].append(time.time() - consts.start_time)

                # random selection
                set_size = 200
                indexes = np.random.choice(len(mc), set_size)
                q = np.copy(q[indexes])
                mc = np.copy(mc[indexes])

                score = np.array(scores)
                frames = np.array(frames)

                player_params['score'] = score.mean()
                # player_params['high'] = score.max()
                player_params['frames'] = np.percentile(frames, 90)
                player_params['high'] = np.percentile(scores, 90)

                if args.tensorboard:

                    self.writer.add_scalar('score/%s' % player_name, float(score.mean()), n)
                    self.writer.add_scalar('high/%s' % player_name, float(score.max()), n)
                    self.writer.add_scalar('low/%s' % player_name, float(score.min()), n)
                    self.writer.add_scalar('std/%s' % player_name, float(score.std()), n)
                    self.writer.add_scalar('frames/%s' % player_name, float(frames.mean()), n)

                    try:
                        self.writer.add_histogram("mc/%s" % player_name, mc, n, 'fd')
                        self.writer.add_histogram("q/%s" % player_name, q, n, 'fd')
                    except:
                        pass

                np.save(results_filename, results)

                if self.log_scores:
                    logger.info("Save NPY file: %d_%s_%d_%s.npy" % (n, uuid, kk, player_name))
                    stat_filename = os.path.join(scores_dir, "%d_%s_%d_%s" % (n, uuid, kk, player_name))
                    np.save(stat_filename, stats)

                kk += 1

    def print_actions_statistics(self, a_agent, a_player, n, Hbeta, Hpi, adv_a, q_a, r_mc):

        # print action meanings
        logger.info("Actions statistics: \tH(beta) = %g |\t H(pi) = %g |" % (Hbeta, Hpi))
        action_space = len(self.action_meanings)

        line = ''
        line += "|\tActions Names\t"
        for a in self.action_meanings:
            line += "|%s%s  " % (a[:11], ' '*(11 - len(a[:11])))
        line += "|"
        logger.info(line)

        n_actions = len(a_agent)
        applied_player_actions = (np.bincount(np.concatenate((a_player, np.arange(action_space)))) - 1) / n_actions
        applied_agent_actions = (np.bincount(np.concatenate((a_agent, np.arange(action_space)))) - 1) / n_actions

        line = ''
        line += "|\tPlayer actions\t"
        for a in applied_player_actions:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

        line = ''
        line += "|\tAgent actions\t"
        for a in applied_agent_actions:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

        match_precentage_by_action = []
        error_precentage_by_action = []
        adv_by_action = []
        q_by_action = []
        r_by_action = []
        for a in range(action_space):

            n_action_player = (a_player == a).sum()
            if n_action_player:
                match_precentage_by_action.append((a_agent[a_player == a] == a).sum() / n_action_player)
            else:
                match_precentage_by_action.append(-0.01)

            if n_action_player:
                adv_by_action.append((adv_a[a_player == a]).sum() / n_action_player)
            else:
                adv_by_action.append(-0.01)

            if n_action_player:
                q_by_action.append((q_a[a_player == a]).sum() / n_action_player)
            else:
                q_by_action.append(-0.01)

            if n_action_player:
                r_by_action.append((r_mc[a_player == a]).sum() / n_action_player)
            else:
                r_by_action.append(-0.01)

            n_not_action_player = (a_player != a).sum()
            if n_not_action_player:
                error_precentage_by_action.append((a_agent[a_player != a] == a).sum() / n_not_action_player)
            else:
                error_precentage_by_action.append(-0.01)

        line = ''
        line += "|\tAdvantage    \t"
        for a in adv_by_action:
            line += "|%.2f\t    " % a
        line += "|"
        logger.info(line)

        line = ''
        line += "|\tQ(s,a)       \t"
        for a in q_by_action:
            line += "|%.2f\t    " % a
        line += "|"
        logger.info(line)

        line = ''
        line += "|\tR Monte-Carlo\t"
        for a in r_by_action:
            line += "|%.2f\t    " % a
        line += "|"
        logger.info(line)


        line = ''
        line += "|\tMatch by action\t"
        for a in match_precentage_by_action:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

        line = ''
        line += "|\tError by action\t"
        for a in error_precentage_by_action:
            line += "|%.2f\t    " % (a*100)
        line += "|"
        logger.info(line)

    def demonstrate(self, params=None):

        agent = self.choose_agent()(self.replay_dir, player=True, checkpoint=self.checkpoint)

        # load model
        try:
            if params is not None:
                aux = agent.resume(params)
            elif self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError
        except:  # when reading and writing collide
            time.sleep(2)
            if params is not None:
                aux = agent.resume(params)
            elif self.load_last:
                aux = agent.resume(self.checkpoint)
            elif self.load_best:
                aux = agent.resume(self.checkpoint_best)
            else:
                raise NotImplementedError

        player = agent.demonstrate(128)

        for i, step in enumerate(player):
            # print("here %d" % i)
            yield step

            # print("out")