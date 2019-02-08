#!/usr/bin/env bash

identifier=$1
game=$2
resume=$3
aux="${@:4}"

if [ $resume != "new" ]; then
    args="--resume=$3 --load-last-model"
    echo "Resume Experiment: $identifier $3"
else
    args=""
    echo "New Experiment"
fi

CUDA_VISIBLE_DEVICES=0, python main.py --learn --identifier=$identifier --game=$game $args $aux --algorithm=rbi \
--n-steps=4 --no-reward-shape --no-dropout --no-infinite-horizon --target=td \
--batch=128 --hidden-features=512 --clip=1 --discount=0.99 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.6 --priority-beta=0.4 --epsilon-a=0.001 \
--epsilon=0.00164 --cpu-workers=48 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=5000 \
--random-initialization=2500 --player-replay-size=2500 --update-memory-interval=100 \
--load-memory-interval=100 --replay-updates-interval=2500 --replay-memory-size=2000000 --n-actors=16
