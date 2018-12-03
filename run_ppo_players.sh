#!/usr/bin/env bash

identifier=$1
resume=$2
game=$3


CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=0 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=1 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=2 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=3 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=4 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=5 &

CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=6 &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=7 &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=8 &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=9 &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=10 &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=11 &

CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=12 &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=13 &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=14 &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --n-actors=16 --actor-index=15 &

CUDA_VISIBLE_DEVICES=0, python main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo &
CUDA_VISIBLE_DEVICES=3, python main.py --choose --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --play-episodes-interval=16 --wait=150 &
CUDA_VISIBLE_DEVICES=3, python main.py --choose --identifier=$identifier --resume=$resume --load-last-model --game=$game --algorithm=ppo --cuda-default=0 --play-episodes-interval=16 --wait=300 &

