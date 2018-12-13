#!/usr/bin/env bash

identifier=$1
resume=$2
game=$3

CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=0 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=1 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=2 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=3 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=4 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=5 --n-steps=3 --clip=1 --no-reward-shape  &

CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=6 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=7 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=8 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=9 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=10 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=11 --n-steps=3 --clip=1 --no-reward-shape  &

CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=12 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=13 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=14 --n-steps=3 --clip=1 --no-reward-shape  &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=15 --n-steps=3 --clip=1 --no-reward-shape  &

CUDA_VISIBLE_DEVICES=0, python main.py --clean --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game &
CUDA_VISIBLE_DEVICES=3, python main.py --choose --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --play-episodes-interval=16 --wait=150 --epsilon-pre=0.00164 --epsilon-post=0.00164 --no-tensorboard &
CUDA_VISIBLE_DEVICES=3, python main.py --choose --algorithm=ape --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --play-episodes-interval=16 --wait=300 --epsilon-pre=0.00164 --epsilon-post=0.00164 --no-tensorboard &