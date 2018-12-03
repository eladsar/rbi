import numpy as np
import torch
import os
import parse
from tqdm import tqdm
import cv2
import itertools
import fcntl

from config import args, consts

# local consts

img_width = args.width
img_height = args.height

# consts:

# cv2.NORM_L2: 4
# cv2.CV_32F: 5
# cv2.COLOR_BGR2GRAY

img_dtype = cv2.CV_32F
img_norm = cv2.NORM_MINMAX
img_bgr2gray = cv2.COLOR_BGR2GRAY
img_rgb2gray = cv2.COLOR_RGB2GRAY
img_gray2rgb = cv2.COLOR_GRAY2RGB
img_bgr2rgb = cv2.COLOR_BGR2RGB
img_rgb2bgr = cv2.COLOR_RGB2BGR
interpolation =  cv2.INTER_LINEAR  # cv2.INTER_AREA  #
imread_grayscale = cv2.IMREAD_GRAYSCALE

img_threshold = cv2.THRESH_BINARY
img_inter = cv2.INTER_NEAREST

clip = args.clip
clip_rho = args.clip_rho

infinite_horizon = args.infinite_horizon

r_scale = consts.scale_reward[args.game]
friction = args.friction_reward
termination_reward = args.termination_reward

# if clip:
#
# else:
#     termination_reward = args.termination_reward * r_scale


def convert_screen_to_rgb(img, resize=False):
    img = cv2.cvtColor(img.numpy(), img_gray2rgb)
    #
    if resize:
        img = img / img.max()
        img = cv2.resize(img, (128, 1024), interpolation=img_inter)
    return torch.from_numpy(np.rollaxis(img, 2, 0))


# def _h_np(r):
#     return np.sign(r) * np.log(1 + np.abs(r))
#
#
# def _hinv_np(r):
#     return np.sign(r) * (np.exp(np.abs(r)) - 1)
#
# def _hinv_np_2tag(r):
#     return np.sign(r) * np.exp(np.abs(r))
#
#
# def _h_torch(r):
#     return torch.sign(r) * torch.log(1 + torch.abs(r))
#
#
# def _hinv_torch(r):
#     return torch.sign(r) * (torch.exp(torch.abs(r)) - 1)


def _h_np(r):
    return np.sign(r) * (np.sqrt(np.abs(r) + 1) - 1) + 0.01 * r


def _hinv_np(r):
    return np.sign(r) * (((np.sqrt(1 + 0.04 * (np.abs(r) + 1.01)) - 1) / 0.02) ** 2 - 1)


def _h_torch(r):
    return torch.sign(r) * (torch.sqrt(torch.abs(r) + 1) - 1) + 0.01 * r


def _hinv_torch(r):
    return torch.sign(r) * (((torch.sqrt(1 + 0.04 * (torch.abs(r) + 1.01)) - 1) / 0.02) ** 2 - 1)
#
# def _hinv_np_tag(r):
#     raise NotImplementedError
#
# def _hinv_torch_tag(r):
#     return 100 - 5000 / torch.sqrt(100 * torch.abs(r) + 2601)

# def _h_np(r):
#     a = np.abs(r)
#     piece = (a <= 1)
#     return np.sign(r) * (a * piece + (2 * np.sqrt(np.maximum(a, 0)) - 1) * (1 - piece))
#
#
# def _hinv_np(r):
#     a = np.abs(r)
#     piece = (a <= 1)
#     return np.sign(r) * (a * piece + (0.25 * a ** 2 + 0.5 * a + 0.25) * (1 - piece))
#
#
# def _h_torch(r):
#     a = torch.abs(r)
#     piece = (a <= 1).float()
#     return torch.sign(r) * (a * piece + (2 * torch.sqrt(torch.clamp(a, min=0)) - 1) * (1 - piece))
#
#
# def _hinv_torch(r):
#     a = torch.abs(r)
#     piece = (a <= 1).float()
#     return torch.sign(r) * (a * piece + (0.25 * a ** 2 + 0.5 * a + 0.25) * (1 - piece))


# def _h_np(r):
#     a = np.abs(r)
#     piece = (a <= 9)
#     return np.sign(r) * (a * piece + (7 + 2 * np.sqrt(np.maximum(a - 8, 0))) * (1 - piece))
#
#
# def _hinv_np(r):
#     a = np.abs(r)
#     piece = (a <= 9)
#     return np.sign(r) * (a * piece + (a ** 2 - 14 * a + 81) / 4 * (1 - piece))
#
#
# def _h_torch(r):
#     a = torch.abs(r)
#     piece = (a <= 9).float()
#     return torch.sign(r) * (a * piece + (7 + 2 * torch.sqrt(torch.clamp(a - 8, min=0))) * (1 - piece))
#
#
# def _hinv_torch(r):
#     a = torch.abs(r)
#     piece = (a <= 9).float()
#     return torch.sign(r) * (a * piece + (a ** 2 - 14 * a + 81) / 4 * (1 - piece))

def _hinv_np_tag(r):
    raise NotImplementedError


def _hinv_torch_tag(r):
    raise NotImplementedError


def _idle(r):
    return r


if args.reward_shape:
    h_np = _h_np
    hinv_np = _hinv_np
    h_torch = _h_torch
    hinv_torch = _hinv_torch
    hinv_np_tag = _hinv_np_tag
    hinv_torch_tag = _hinv_torch_tag
else:
    h_np = _idle
    hinv_np = _idle
    h_torch = _idle
    hinv_torch = _idle
    hinv_np_tag = _idle
    hinv_torch_tag = _idle


def get_mc_value(rewards, discount):

    if infinite_horizon:
        rewards = [list(itertools.chain(*rewards))]

    lives = len(rewards)

    values = []
    for life in range(lives):

        if not len(rewards[life]):
            continue
        discounts = discount ** np.arange(len(rewards[life])+1)

        r = np.array(rewards[life])
        if clip > 0:
            r = np.clip(r, -clip, clip)
        else:
            r = r / r_scale

        r[-1] += termination_reward
        val = np.zeros(r.shape)
        for i in range(len(r)):
            val[i] = (r[i:] * discounts[:-i-1]).sum()

        values.append(val)

    return np.concatenate(values).astype(np.float32)


def get_td_value(rewards, v_target, discount, n_steps):

    lives = len(rewards)

    values = []
    for life in range(lives):

        episode_len = len(rewards[life])

        if not episode_len:
            continue

        discounts = discount ** np.arange(n_steps)

        r = np.array(rewards[life], dtype=np.float64)
        v_t = np.concatenate((np.array(v_target[life], dtype=np.float64), np.zeros(n_steps)))

        r = np.clip(r, -clip, clip)
        r[-1] += termination_reward

        val = np.correlate(r, discounts, mode="full")[n_steps-1:]
        val += discount ** n_steps * v_t[n_steps:]

        values.append(val)

    return np.concatenate(values).astype(np.float32)


def get_gae_est(rewards, v_target, discount):

    lives = len(rewards)
    gamma = 0.95

    values = []
    for life in range(lives):

        episode_len = len(rewards[life])

        if not episode_len:
            continue

        discounts = (gamma * discount) ** np.arange(episode_len)
        r = np.array(rewards[life], dtype=np.float64)
        v_t = np.concatenate((np.array(v_target[life], dtype=np.float64), np.zeros(1)))
        r = np.clip(r, -clip, clip)
        r[-1] += termination_reward

        delta = r + v_t[1:] - discount * v_t[:-1]

        val = np.correlate(delta, discounts, mode="full")[episode_len-1:]
        values.append(val)

    return np.concatenate(values).astype(np.float32)


def get_rho_is(rho, n_steps):

    lives = len(rho)

    rho_is = []
    for life in range(lives):

        episode_len = len(rho[life])

        if not episode_len:
            continue

        v_t = np.concatenate((np.array(rho[life], dtype=np.float64), np.ones(n_steps)))

        val = np.ones(episode_len)
        for i in range(1, n_steps):
            val *= v_t[i:episode_len+i]

        rho_is.append(val)

    return np.minimum(np.concatenate(rho_is), clip_rho).astype(np.float32)


# def get_td_value(rewards, v_target, discount, n_steps):
#
#     if infinite_horizon:
#         rewards = [list(itertools.chain(*rewards))]
#         v_target = [list(itertools.chain(*v_target))]
#
#     lives = len(rewards)
#
#     values = []
#     for life in range(lives):
#
#         episode_len = len(rewards[life])
#
#         if not episode_len:
#             continue
#
#         discounts = discount ** np.arange(len(rewards[life])+1)
#
#         r = np.array(rewards[life], dtype=np.float64)
#         v_t = np.array(v_target[life], dtype=np.float64)
#
#         if clip > 0:
#             r = np.clip(r, -clip, clip)
#         else:
#             r = r / r_scale
#             # r = np.sign(r) * np.log(np.abs(r) + 1)
#
#         r[-1] += termination_reward
#         val = np.zeros(r.shape)
#         for i in range(len(r)):
#             val[i] = (r[i:] * discounts[:-i-1]).sum()
#
#         if episode_len > n_steps:
#             val_shift = discount ** n_steps * np.concatenate((val[n_steps:], np.zeros(n_steps)))
#             v_t_shift = discount ** n_steps * np.concatenate((v_t[n_steps:], np.zeros(n_steps)))
#             val = val - val_shift + v_t_shift
#
#         values.append(val)
#
#     return np.concatenate(values).astype(np.float32)


# def get_rho_is(rho, n_steps):
#
#     if infinite_horizon:
#         rho = [list(itertools.chain(*rho))]
#
#     lives = len(rho)
#
#     rho_is = []
#     for life in range(lives):
#
#         episode_len = len(rho[life])
#
#         if not episode_len:
#             continue
#
#         v_t = np.array(rho[life], dtype=np.float64)
#         val = np.zeros(v_t.shape)
#         v_t = np.concatenate((v_t, np.ones(n_steps)))
#
#         val[0] = np.prod(v_t[1:n_steps])
#         for i in range(1, len(val)):
#             val[i] = val[i - 1] / v_t[i] * v_t[i + n_steps]
#
#         rho_is.append(val)
#
#     return np.minimum(np.concatenate(rho_is), clip_rho).astype(np.float32)


def get_tde_value(rewards, discount, n_steps):

    if infinite_horizon:
        rewards = [list(itertools.chain(*rewards))]

    lives = len(rewards)

    values = []
    terminals = []
    for life in range(lives):

        episode_len = len(rewards[life])

        if not episode_len:
            continue

        discounts = discount ** np.arange(len(rewards[life])+1)

        r = np.array(rewards[life], dtype=np.float64)

        if clip > 0:
            r = np.clip(r, -clip, clip)
        else:
            r = r / r_scale
            # r = np.sign(r) * np.log(np.abs(r) + 1)

        r[-1] += termination_reward
        val = np.zeros(r.shape)
        t = np.zeros(r.shape)
        t[-n_steps:] = 1

        for i in range(len(r)):
            val[i] = (r[i:] * discounts[:-i-1]).sum()

        if episode_len > n_steps:
            val_shift = discount ** n_steps * np.concatenate((val[n_steps:], np.zeros(n_steps)))
            val = val - val_shift

        values.append(val)
        terminals.append(t)

    return np.concatenate(values).astype(np.float32), np.concatenate(terminals).astype(np.float32)


def get_gtd_value(rewards, v_target, discount, mu, sigma):

    if infinite_horizon:
        rewards = [list(itertools.chain(*rewards))]

    lives = len(rewards)

    values = []
    for life in range(lives):

        episode_len = len(rewards[life])

        if not episode_len:
            continue

        x_axis = np.arange(len(rewards[life])+2)
        discounts = discount ** x_axis

        alpha = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x_axis - mu) ** 2 / (2 * sigma ** 2))
        alpha_sum_k_inf = np.flipud(np.cumsum(np.flipud(alpha)))

        r = np.array(rewards[life], dtype=np.float64)
        v_t = np.array(v_target[life], dtype=np.float64)

        if clip > 0:
            r = np.clip(r, -clip, clip)
        else:
            r = r / r_scale
            # r = np.sign(r) * np.log(np.abs(r) + 1)

        r[-1] += termination_reward
        val = np.zeros(r.shape)
        for i in range(len(r)):
            scale = np.clip((alpha_sum_k_inf[0] - alpha_sum_k_inf[-i-1]), a_min=1e-6, a_max=None)
            val[i] = ((r[i:] * discounts[:-i-2] * (alpha_sum_k_inf[:-i-2] - alpha_sum_k_inf[-i-1])).sum() +
                      (alpha[1:-i-2] * discounts[1:-i-2] * v_t[i+1:]).sum()) / scale

        values.append(val)

    return np.concatenate(values).astype(np.float32)


def lock_file(file):

    fo = open(file, "r+b")
    while True:
        try:
            fcntl.lockf(fo, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError:
            pass

    return fo


def release_file(fo):
    fcntl.lockf(fo, fcntl.LOCK_UN)
    fo.close()