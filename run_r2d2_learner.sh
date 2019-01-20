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

CUDA_VISIBLE_DEVICES=0,1, python main.py --learn --identifier=$identifier --game=$game $args $aux --algorithm=r2d2 \
--n-steps=5 --no-reward-shape --no-dropout --infinite-horizon --target=tde \
--batch=32 --seq-length=30 --burn-in=10 --seq-overlap=20 \
--hidden-features=512 --hidden-features-rnn=512 --clip=0 --discount=0.997 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.9 --priority-beta=0.6 --epsilon-a=0.001 \
--epsilon-pre=0.00164 --epsilon-post=0.00164 --cpu-workers=48 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=1000 \
--random-initialization=1000 --player-replay-size=1000 --update-memory-interval=20 \
--load-memory-interval=100 --replay-updates-interval=500 --replay-memory-size=4000000
