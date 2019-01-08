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

python main.py --learn --identifier=$identifier --game=$game $args $aux --algorithm=r2d2 \
--n-steps=3 --no-reward-shape --no-dropout --no-infinite-horizon --target=tde \
--batch=64 --seq-length=1 --burn-in=1 --seq-overlap=1 \
--hidden-features=512 --hidden-features-rnn=512 --clip=1 --discount=0.99 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.9 --epsilon-a=0.001 \
--epsilon-pre=0.00164 --epsilon-post=0.00164 --cpu-workers=24 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=5000 \
--random-initialization=2500 --player-replay-size=1000 --update-memory-interval=100 \
--load-memory-interval=250 --replay-updates-interval=1000 --replay-memory-size=2000000
