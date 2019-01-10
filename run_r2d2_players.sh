#!/usr/bin/env bash

identifier=$1
resume=$2
game=$3

args="--n-steps=5 --reward-shape --no-dropout --infinite-horizon --target=tde \
--batch=32 --seq-length=30 --burn-in=10 --seq-overlap=10 \
--hidden-features=128 --hidden-features-rnn=128 --clip=0 --discount=0.997 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.9 --priority-beta=0.6 --epsilon-a=0.001 \
--epsilon-pre=0.00164 --epsilon-post=0.00164 --cpu-workers=48 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=1000 \
--random-initialization=1000 --player-replay-size=1000 --update-memory-interval=20 \
--load-memory-interval=100 --replay-updates-interval=500 --replay-memory-size=4000000 \
--n-players=16"

CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=0 $args  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=1 $args  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=2 $args  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=3 $args  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=4 $args  &
CUDA_VISIBLE_DEVICES=1, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=5 $args  &

CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=6 $args  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=7 $args  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=8 $args  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=9 $args  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=10 $args  &
CUDA_VISIBLE_DEVICES=2, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=11 $args  &

CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=12 $args  &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=13 $args  &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=14 $args  &
CUDA_VISIBLE_DEVICES=3, python main.py --multiplay --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --n-actors=16 --actor-index=15 $args  &

CUDA_VISIBLE_DEVICES=0, python main.py --clean --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game &
CUDA_VISIBLE_DEVICES=3, python main.py --choose --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --play-episodes-interval=16 --wait=150 $args  &
CUDA_VISIBLE_DEVICES=3, python main.py --choose --algorithm=r2d2 --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=0 --play-episodes-interval=16 --wait=300 $args  &