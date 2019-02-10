#!/usr/bin/env bash

identifier=$1
game=$2
resume=$3

loc=`dirname "%0"`

if [ $game = "berzerk" ]; then
    tensor="--no-tensorboard"
fi
if [ $game = "frostbite" ]; then
    tensor="--no-tensorboard"
fi
if [ $game = "breakout" ]; then
    tensor="--no-tensorboard"
fi

args="--algorithm=ape --n-steps=3 --no-reward-shape --no-dropout --no-infinite-horizon \
--batch=128 --hidden-features=512 --clip=1 --discount=0.99 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.6 --priority-beta=0.4 --epsilon-a=0.001 \
--epsilon=0.00164 --cpu-workers=48 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=5000 \
--random-initialization=2500 --player-replay-size=2500 --update-memory-interval=100 \
--load-memory-interval=100 --replay-updates-interval=2500 --replay-memory-size=2000000 --n-actors=16"

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=0 $args &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=1 $args &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=2 $args &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=3 $args &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=4 $args &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=5 $args &

CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=6 $args &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=7 $args &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=8 $args &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=9 $args &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=10 $args &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=11 $args &

CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=12 $args &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=13 $args &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=14 $args &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=15 $args &

CUDA_VISIBLE_DEVICES=3, python $loc/main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game $args &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --evaluate --identifier=$identifier --resume=$resume --load-last-model --game=$game --play-episodes-interval=16 --wait=150 --epsilon=0.00164  $args $tensor &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --evaluate --identifier=$identifier --resume=$resume --load-last-model --game=$game --play-episodes-interval=16 --wait=300 --epsilon=0.00164  $args $tensor &