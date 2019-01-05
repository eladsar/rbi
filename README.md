# rbi
Implementation of distributed RL algorithms:

Baselines:
1. Ape-X (https://arxiv.org/abs/1803.00933)
2. R2D2 (https://openreview.net/pdf?id=r1lyTjAqYX)
3. PPO (https://arxiv.org/abs/1707.06347)

RBI:
A safe reinforcement learning algorithm 

Currently, supported environment is ALE
## How to run

### Run Learner:

CUDA_VISIBLE_DEVICES=0, python main.py --learn --identifier=spaceinvaders_debug --game=spaceinvaders
CUDA_VISIBLE_DEVICES=0, python main.py --learn --identifier=spaceinvaders_ape --game=spaceinvaders --algorithm=ape --n-steps=3 --no-reward-shape --clip=1
CUDA_VISIBLE_DEVICES=0, python main.py --learn --identifier=final_ape_spaceinvaders --game=spaceinvaders --algorithm=ape --n-steps=3 --no-reward-shape --clip=1
CUDA_VISIBLE_DEVICES=0, python main.py --learn --identifier=final_ppo_spaceinvaders --game=spaceinvaders --algorithm=ppo

### Run Actors:

sh run_rbi_players.sh debug 0 spaceinvaders

### Run Evaluation player:

CUDA_VISIBLE_DEVICES=0, python main.py --play --identifier=final_rbi_spaceinvaders --game=spaceinvaders --resume=0 --load-last-model --play-episodes-interval=120 --no-frame-limit


