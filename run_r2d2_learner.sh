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
--batch=32 --seq-length=30 --burn-in=10 --seq-overlap=10 \
--hidden-features=512 --hidden-features-rnn=512 --clip=1 --discount=0.99 \
--termination-reward=-1 --friction-reward=0 --priority-alpha=0.5 --epsilon-a=0.001 \
--epsilon-pre=0.00164 --epsilon-post=0.00164 --cpu-workers=24 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=1000 \
--random-initialization=1000 --player-replay-size=1000 --update-memory-interval=50 \
--load-memory-interval=150 --replay-updates-interval=500 --replay-memory-size=2000000
