import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.interpolate as interp
from matplotlib.ticker import FuncFormatter

run_time = time.strftime("%y%m%d_%H%M%S")

root_dir = os.path.abspath("/home/elad/Dropbox/projects/deeprl/results/final_rbi/rbi/")

reroute_experiment_dict = {"breakout": "breakout_no_mix_exp_0000_20180828_183917",
                           "enduro": "enduro_no_is_exp_0000_20180806_082648",
                           "freeway": "freeway_no_is_exp_0000_20180805_091920",
                           # "icehockey": "icehockey_no_is_exp_0000_20180806_080719",
                           "icehockey": "icehockey_m_3p8_exp_0001_20180827_135203",
                           "kangaroo": "kangaroo_no_is_exp_0000_20180806_212312",
                           "mspacman": "mspacman_no_mix_exp_0000_20180828_163722",
                           "qbert": "qbert_m_3p8_exp_0001_20180827_135148",
                           # "seaquest": "seaquest_no_is_exp_0001_20180807_081124",
                           "seaquest": "seaquest_no_mix_exp_0001_20180828_093138",
                           "spaceinvaders": "spaceinvaders_no_is_exp_0002_20180805_215111",
                           # "berzerk": "berzerk_no_is_exp_0001_20180809_093943",
                           "berzerk": "berzerk_v_scale_exp_0000_20180825_213327",
                           "asterix": "asterix_no_is_exp_0000_20180808_185200",
                           "frostbite": "frostbite_no_is_exp_0000_20180808_184623",
                           }



ape_experiment_dict = {"breakout": "breakout_ape_exp_0003_20180811_202933",
                       "enduro": "enduro_ape_exp_0000_20180814_032513",
                       "freeway": "freeway_ape_exp_0000_20180813_085028",
                       "icehockey": "icehockey_ape_exp_0000_20180811_064621",
                       "kangaroo": "kangaroo_ape_exp_0000_20180812_143246",
                       "mspacman": "mspacman_ape_exp_0000_20180812_162120",
                       "qbert": "qbert_ape_exp_0002_20180810_103813",
                       # "qbert": "qbert_boltz_expape_exp_0002_20180821_230743",
                       "seaquest": "seaquest_ape_exp_0001_20180811_011410",
                       "spaceinvaders": "spaceinvaders_ape_exp_0019_20180811_205801",
                       "berzerk": "berzerk_ape_exp_0000_20180814_032425",
                        # "berzerk": "berzerk_boltz_expape_exp_0001_20180821_231355",
                       "asterix": "asterix_ape_exp_0000_20180809_175422",
                       "frostbite": "frostbite_ape_exp_0000_20180813_081751",
                       }

# reroute_mini_dict = {"mspacman": "mspacman_no_mix_exp_0000_20180828_163722",
#                        "qbert": "qbert_m_3p8_exp_0001_20180827_135148",
#                        "spaceinvaders": "spaceinvaders_no_is_exp_0002_20180805_215111",
#                        # "berzerk": "berzerk_v_scale_exp_0000_20180825_213327",
#                        "icehockey": "icehockey_m_3p8_exp_0001_20180827_135203",
#                        }
#
#
#
#
# ape_mini_dict = {      "mspacman": "mspacman_ape_exp_0000_20180812_162120",
#                        "qbert": "qbert_ape_exp_0002_20180810_103813",
#                        "spaceinvaders": "spaceinvaders_ape_exp_0019_20180811_205801",
#                        # "berzerk": "berzerk_ape_exp_0000_20180814_032425",
#                        "icehockey": "icehockey_ape_exp_0000_20180811_064621",
#                        }


reroute_mini_dict = {"mspacman": "mspacman_uniform_is_exp_0000_20190213_195016",
                       "qbert": "qbert_uniform_is_exp_0000_20190212_090734",
                       "spaceinvaders": "spaceinvaders_rbi_uniform_exp_0000_20190215_075454",
                       "icehockey": "icehockey_m_3p8_exp_0001_20180827_135203",
                       }




ape_mini_dict = {      "mspacman": "mspacman_ape_exp_0001_20190213_085338",
                       "qbert": "qbert_ape_huber_exp_0001_20190211_222400",
                       "spaceinvaders": "spaceinvaders_ape_exp_0000_20190215_002315",
                       "icehockey": "icehockey_ape_exp_0000_20180811_064621",
                       }


ablation_dict = {
    "qbert": {
        "rbi": "qbert_m_3p8_exp_0001_20180827_135148",
        "nois": "qbert_ablation_is_exp_0000_20180902_080000",
        "noprior": "qbert_ablation_priority_exp_0001_20180830_223615",
        "noexp": "qbert_ablation_explore_exp_0000_20180901_012300",
        "ppo": "qbert_ablation_greedy_exp_0002_20180904_173713",
    },

    "mspacman": {
        "rbi": "mspacman_no_mix_exp_0000_20180828_163722",
        "nois": "mspacman_ablation_nois_exp_0000_20180902_112034",
        "noprior": "mspacman_ablation_noprior_exp_0000_20180902_112712",
        "noexp": "mspacman_ablation_noexp_exp_0000_20180902_112320",
        "ppo": "mspacman_ablation_greedy_exp_0000_20180904_173821",
    }
}


def convert_to_dataframe(experiment):

    run_dir = os.path.join(root_dir, experiment, "scores")
    save_dir = os.path.join(root_dir, experiment, "postprocess")

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    elif os.path.isfile(os.path.join(save_dir, "df_reroute")) and os.path.isfile(os.path.join(save_dir, "df_behavioral")):
        return

    results_reroute = {'score': [], 'frame': [], 'n': [], 'time': []}
    results_behavioral = {'score': [], 'frame': [], 'n': [], 'time': []}

    for d in os.listdir(run_dir):
        print(d)
        for f in os.listdir(os.path.join(run_dir, d)):

            if "behavioral" in f:
                for key in results_behavioral:
                    item = np.load(os.path.join(run_dir, d, f)).item()
                    results_behavioral[key] += item[key]
            else:
                for key in results_reroute:
                    item = np.load(os.path.join(run_dir, d, f)).item()
                    results_reroute[key] += item[key]

    df_reroute = pd.DataFrame(results_reroute)
    df_behavioral = pd.DataFrame(results_behavioral)

    df_reroute.to_pickle(os.path.join(save_dir, "df_reroute"))
    df_behavioral.to_pickle(os.path.join(save_dir, "df_behavioral"))


def preprocess(df):

    idx = pd.RangeIndex(0, 3125000, 100)

    df = df[df['n'] <= 3125000]

    df = df.sort_values("n")
    df = df.groupby("n").mean()
    df = df.reset_index().set_index("n")
    df.reindex(idx)
    df = df.interpolate()
    df = df.rolling(300, win_type='blackmanharris', min_periods=1, center=True).mean()
    return df


def preprocess_quantiles(df):

    idx = pd.RangeIndex(0, 3125000, 100)

    df = df[df['n'] <= 3125000]

    df = df.sort_values("n")
    df = df.groupby("n").mean()
    df = df.reset_index()
    df = df.set_index("n")
    df.reindex(idx)
    df = df.interpolate()

    # df_u = df_u.interpolate()
    df_u = df.rolling(100, min_periods=1, center=True).quantile(0.75, interpolation='linear')
    # df_u = df.rolling(10, win_type='blackmanharris', min_periods=1, center=True).quantile(0.75, interpolation='linear')

    # df_d = df_d.interpolate()
    # df_d = df.rolling(10, win_type='blackmanharris', min_periods=1, center=True).quantile(0.25, interpolation='linear')
    df_d = df.rolling(100, min_periods=1, center=True).quantile(0.25, interpolation='linear')


    return df_u, df_d


def cumulative(df):

    idx = pd.RangeIndex(0, 3125000, 100)
    df = df.sort_values("n")
    df = df.reset_index()
    df.score = df.score.cumsum() / np.arange(len(df.score))
    df = df.groupby("n").mean().reset_index()
    df = df.set_index("n")
    df.reindex(idx)
    df = df.interpolate()
    df = df.rolling(200, win_type='blackmanharris', min_periods=1, center=True).mean()
    return df


def timeprocess(df):

    time = (df.time - df.time.min()) / 3600
    score = df.score

    f = interp.interp1d(time, score)

    time_indx = np.arange(0, 24, 1/60)

    df = df.set_index("time")

    return df


def plot_cumulative():

    plt.style.use('ggplot')
    plt.figure(1, figsize=(12, 4))
    plt.rc('ytick', labelsize=6)
    plt.rc('xtick', labelsize=6)

    formatter = FuncFormatter(millions)

    i = 0
    for experiment in sorted(reroute_experiment_dict.keys()):

        i += 1
        if reroute_experiment_dict[experiment] == "na":
            continue

        # load reroute DF
        save_dir = os.path.join(root_dir, reroute_experiment_dict[experiment], "postprocess")
        ape_dir = os.path.join(root_dir, ape_experiment_dict[experiment], "postprocess")

        df_reroute = pd.read_pickle(os.path.join(save_dir, "df_reroute"))
        df_ape = pd.read_pickle(os.path.join(ape_dir, "df_reroute"))

        df_reroute = cumulative(df_reroute)
        df_ape = cumulative(df_ape)
        plt.subplot(2, 6, i)

        x1 = df_reroute.index
        y1 = df_reroute.score

        x2 = df_ape.index
        y2 = df_ape.score
        # y = df_reroute.score.rolling(100, win_type='triang', min_periods=1, center=True).mean()

        std = df_reroute.score.rolling(100).std()

        ax = plt.gca()
        # plt.plot(x2, y2, x1, y1, label=experiment)
        plt.plot(x2, y2, label="Ape-X")
        plt.plot(x1, y1, label="RBI")
        # plt.plot(x, std / y.abs(), label=experiment)
        plt.title(experiment, fontsize=8)
        if i > 6:
            plt.xticks([0, 1e6, 2e6, 3e6], ["0", "1M", "2M", "3M"])
        else:
            plt.xticks([0, 1e6, 2e6, 3e6], ["", "", "", ""])

        if i == 6:
            plt.legend()
        ax.yaxis.set_major_formatter(formatter)

    # plt.show()
    plt.savefig("/home/elad/Dropbox/projects/deeprl/results/final_rbi/plots/cumsum_%s.pdf" % run_time, bbox_inches="tight")


def plot_mini_charts():

    plt.style.use('ggplot')
    plt.figure(1, figsize=(18, 3))
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=10)

    formatter = FuncFormatter(millions)

    i = 0
    for experiment in sorted(reroute_mini_dict.keys()):

        i += 1
        if reroute_mini_dict[experiment] == "na":
            continue

        # load reroute DF
        save_dir = os.path.join(root_dir, reroute_mini_dict[experiment], "postprocess")
        ape_dir = os.path.join(root_dir, ape_mini_dict[experiment], "postprocess")

        df_reroute = pd.read_pickle(os.path.join(save_dir, "df_reroute"))
        df_ape = pd.read_pickle(os.path.join(ape_dir, "df_reroute"))

        df_reroute_u, df_reroute_d = preprocess_quantiles(df_reroute)
        df_ape_u, df_ape_d = preprocess_quantiles(df_ape)

        df_reroute = preprocess(df_reroute)
        df_ape = preprocess(df_ape)
        # plt.subplot(2, 2, i)
        plt.subplot(1, 4, i)

        x1 = df_reroute.index
        y1 = df_reroute.score

        y1_u = df_reroute_u.score
        y1_d = df_reroute_d.score

        x2 = df_ape.index
        y2 = df_ape.score

        y2_u = df_ape_u.score
        y2_d = df_ape_d.score

        # plt.plot(x2, y2, x1, y1, label=experiment)
        plt.plot(x2, y2, label="Ape-X")
        ax = plt.gca()
        ax.fill_between(x2, y2_u, y2_d, alpha=0.5)
        plt.plot(x1, y1, label="RBI")
        ax = plt.gca()
        ax.fill_between(x1, y1_u, y1_d, alpha=0.5)
        # plt.plot(x, std / y.abs(), label=experiment)
        plt.title(experiment, fontsize=14)

        if i > 0:
            plt.xticks([0, 1e6, 2e6, 3e6], ["0", "1M", "2M", "3M"])
            plt.xlabel("Minibatches (# of backward passes)")
        else:
            plt.xticks([0, 1e6, 2e6, 3e6], ["", "", "", ""])

        if i == 1:
            plt.legend(loc='lower right', prop={'size': 14})
        ax.yaxis.set_major_formatter(formatter)

    # plt.show()
    plt.savefig("/home/elad/Dropbox/projects/deeprl/results/final_rbi/plots/mini_reroute_%s.pdf" % run_time, bbox_inches="tight")


def plot_charts():

    plt.style.use('ggplot')
    plt.figure(1, figsize=(12, 4))
    plt.rc('ytick', labelsize=6)
    plt.rc('xtick', labelsize=6)

    formatter = FuncFormatter(millions)

    i = 0
    for experiment in sorted(reroute_experiment_dict.keys()):

        i += 1
        if reroute_experiment_dict[experiment] == "na":
            continue

        # load reroute DF
        save_dir = os.path.join(root_dir, reroute_experiment_dict[experiment], "postprocess")
        ape_dir = os.path.join(root_dir, ape_experiment_dict[experiment], "postprocess")

        df_reroute = pd.read_pickle(os.path.join(save_dir, "df_reroute"))
        df_ape = pd.read_pickle(os.path.join(ape_dir, "df_reroute"))

        df_reroute_u, df_reroute_d = preprocess_quantiles(df_reroute)
        df_ape_u, df_ape_d = preprocess_quantiles(df_ape)

        df_reroute = preprocess(df_reroute)
        df_ape = preprocess(df_ape)
        plt.subplot(2, 6, i)

        x1 = df_reroute.index
        y1 = df_reroute.score

        y1_u = df_reroute_u.score
        y1_d = df_reroute_d.score

        x2 = df_ape.index
        y2 = df_ape.score

        y2_u = df_ape_u.score
        y2_d = df_ape_d.score

        # plt.plot(x2, y2, x1, y1, label=experiment)
        plt.plot(x2, y2, label="Ape-X")
        ax = plt.gca()
        ax.fill_between(x2, y2_u, y2_d, alpha=0.5)
        plt.plot(x1, y1, label="RBI")
        ax = plt.gca()
        ax.fill_between(x1, y1_u, y1_d, alpha=0.5)
        # plt.plot(x, std / y.abs(), label=experiment)
        plt.title(experiment, fontsize=8)
        if i > 6:
            plt.xticks([0, 1e6, 2e6, 3e6], ["0", "1M", "2M", "3M"])
        else:
            plt.xticks([0, 1e6, 2e6, 3e6], ["", "", "", ""])

        if i == 6:
            plt.legend()
        ax.yaxis.set_major_formatter(formatter)

    # plt.show()
    plt.savefig("/home/elad/Dropbox/projects/deeprl/results/final_rbi/plots/reroute_%s.pdf" % run_time, bbox_inches="tight")


def ablation_test():

    plt.style.use('ggplot')
    plt.figure(1, figsize=(6, 3))
    plt.rc('ytick', labelsize=6)
    plt.rc('xtick', labelsize=6)

    formatter = FuncFormatter(millions)

    i = 0
    for experiment in sorted(ablation_dict.keys()):

        i += 1
        if reroute_experiment_dict[experiment] == "na":
            continue

        # load reroute DF
        rbi_dir = os.path.join(root_dir, ablation_dict[experiment]['rbi'], "postprocess")
        is_dir = os.path.join(root_dir, ablation_dict[experiment]['nois'], "postprocess")
        prior_dir = os.path.join(root_dir, ablation_dict[experiment]['noprior'], "postprocess")
        ppo_dir = os.path.join(root_dir, ablation_dict[experiment]['ppo'], "postprocess")
        explore_dir = os.path.join(root_dir, ablation_dict[experiment]['noexp'], "postprocess")

        df_rbi = pd.read_pickle(os.path.join(rbi_dir, "df_reroute"))
        df_is = pd.read_pickle(os.path.join(is_dir, "df_reroute"))
        df_prior = pd.read_pickle(os.path.join(prior_dir, "df_reroute"))
        df_ppo = pd.read_pickle(os.path.join(ppo_dir, "df_reroute"))
        df_explore = pd.read_pickle(os.path.join(explore_dir, "df_reroute"))

        df_rbi = preprocess(df_rbi)
        df_is = preprocess(df_is)
        df_prior = preprocess(df_prior)
        df_ppo = preprocess(df_ppo)
        df_explore = preprocess(df_explore)

        plt.subplot(1, 2, i)

        x1 = df_rbi.index
        y1 = df_rbi.score

        x2 = df_is.index
        y2 = df_is.score

        x3 = df_prior.index
        y3 = df_prior.score

        x4 = df_ppo.index
        y4 = df_ppo.score

        x5 = df_explore.index
        y5 = df_explore.score

        ax = plt.gca()
        # plt.plot(x2, y2, x1, y1, label=experiment)
        plt.plot(x2, y2, label="No IS Correction", linewidth=1.0)
        plt.plot(x3, y3, label="TDE Priority", linewidth=1.0)
        plt.plot(x5, y5, label="Flat Explotarion", linewidth=1.0)
        plt.plot(x4, y4, label="Greedy", linewidth=1.0, color="goldenrod")
        plt.plot(x1, y1, label="RBI", color="purple")
        # plt.plot(x, std / y.abs(), label=experiment)
        plt.title(experiment, fontsize=8)
        plt.xticks([0, 1e6, 2e6, 3e6], ["0", "1M", "2M", "3M"])

        if i == 2:
            plt.legend()
            plt.legend(prop={'size': 8})
        ax.yaxis.set_major_formatter(formatter)

    # plt.show()
    plt.savefig("/home/elad/Dropbox/projects/deeprl/results/final_rbi/plots/ablation_%s.pdf" % run_time, bbox_inches="tight")


def plot_time():

    plt.style.use('ggplot')
    plt.figure(1, figsize=(12, 4))
    plt.rc('ytick', labelsize=4)
    plt.rc('xtick', labelsize=4)

    i = 0
    for experiment in sorted(reroute_experiment_dict.keys()):

        i += 1
        if reroute_experiment_dict[experiment] == "na":
            continue

        # load reroute DF
        save_dir = os.path.join(root_dir, reroute_experiment_dict[experiment], "postprocess")
        ape_dir = os.path.join(root_dir, ape_experiment_dict[experiment], "postprocess")

        df_reroute = pd.read_pickle(os.path.join(save_dir, "df_reroute"))
        df_ape = pd.read_pickle(os.path.join(ape_dir, "df_reroute"))

        df_reroute = preprocess(df_reroute)
        df_ape = preprocess(df_ape)
        plt.subplot(2, 6, i)

        x1 = (df_reroute.time - df_reroute.time.min()) / 3600
        y1 = df_reroute.score

        x2 = (df_ape.time - df_ape.time.min()) / 3600
        y2 = df_ape.score

        ax = plt.gca()
        plt.plot(x2, y2, x1, y1, label=experiment)
        # plt.plot(x, std / y.abs(), label=experiment)
        plt.title(experiment, fontsize=8)
        # plt.xticks([0, 1e6, 2e6, 3e6], ["0", "1e6", "2e6", "3e6"])

    # plt.show()
    plt.savefig("/home/elad/Dropbox/projects/deeprl/results/final_rbi/plots/time_%s.pdf" % run_time, bbox_inches="tight")


def millions(x, pos):
    'The two args are the value and tick position'

    if x < 1000:
        return '%1.f' % x
    if x < 1e6:
        return '%1.fK' % (x*1e-3)
    else:
        return '%1.fM' % (x*1e-6)


def deterministic_policies():

    import torch

    action_space = 18
    n_steps = 200

    class LinNet(torch.nn.Module):

        def __init__(self):

            super(LinNet, self).__init__()

            self.layers = torch.nn.Sequential(
                torch.nn.Linear(100, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, action_space),
            )

        def forward(self, x):

            y = self.layers(x)
            return y

    net2 = LinNet()
    x = torch.randn(1, 100)
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.001)

    net = LinNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    net.load_state_dict(net2.state_dict())

    # pi1 = torch.autograd.Variable(torch.FloatTensor([1, 0, 0, 0]))

    pi = torch.rand(action_space)
    pi = pi / pi.sum()
    pi = torch.autograd.Variable(pi)
    # pi = torch.autograd.Variable(torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    # pi = torch.autograd.Variable(torch.FloatTensor([1, 0, 0, 0, 0]))
    # pi = torch.autograd.Variable(torch.FloatTensor([0.2, 0.2, 0.2, 0.2, 0.2]))
    # pi = torch.autograd.Variable(torch.FloatTensor([0.9, 0.1, 0., 0., 0.]))
    # pi = torch.autograd.Variable(torch.FloatTensor([0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    pi = torch.autograd.Variable(torch.ones(18)) / 18

    for i in range(n_steps):

        y = net2(torch.autograd.Variable(x))

        # a = int(np.random.choice(action_space, p=pi.data.numpy()))
        #
        # loss = -(torch.nn.functional.log_softmax(y, dim=1)[0, a]).sum()
        loss = -(pi * torch.nn.functional.log_softmax(y, dim=1)).sum()

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

    loss = -(pi * torch.nn.functional.log_softmax(y, dim=1)).sum()
    print(i)
    print("beta: %s" % str(torch.nn.functional.softmax(y, dim=1)))
    print("loss: %g" % loss)

    # pi1 = torch.autograd.Variable(torch.FloatTensor([0, 1, 0, 0]))
    # pi1 = torch.autograd.Variable(torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    for i in range(n_steps):
        y = net(torch.autograd.Variable(x))

        a = int(np.random.choice(action_space, p=pi.data.numpy()))

        loss = -(torch.nn.functional.log_softmax(y, dim=1)[0, a]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = -(pi * torch.nn.functional.log_softmax(y, dim=1)).sum()
    print(i)
    print("beta: %s" % str(torch.nn.functional.softmax(y, dim=1)))
    print("loss: %g" % loss)

    print("pi: %s" % str(pi))


def main():

    # deterministic_policies()

    # for experiment in os.listdir(root_dir):
    #     convert_to_dataframe(experiment)

    # plot_charts()
    plot_mini_charts()
    # plot_time()
    # plot_cumulative()
    # ablation_test()
    print("end of script")


if __name__ == "__main__":
    main()
