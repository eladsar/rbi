#!/usr/bin/env bash

identifier=$1
resume=$2
game=$3

tensor=""

if [ $game = "berzerk" ]; then
    tensor="--no-tensorboard"
fi
if [ $game = "frostbite" ]; then
    tensor="--no-tensorboard"
fi
if [ $game = "breakout" ]; then
    tensor="--no-tensorboard"
fi

CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=22 --actor-index=0 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=22 --actor-index=1 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=22 --actor-index=2 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=22 --actor-index=3 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=22 --actor-index=4 &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=22 --actor-index=5 &


CUDA_VISIBLE_DEVICES=0, python main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game &
CUDA_VISIBLE_DEVICES=0, python main.py --choose --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --play-episodes-interval=16 --wait=150 $tensor &
CUDA_VISIBLE_DEVICES=0, python main.py --choose --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --play-episodes-interval=16 --wait=300 $tensor &
