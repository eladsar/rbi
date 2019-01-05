#!/usr/bin/env bash

identifier=$1
game=$2

resume=$3

if [ $resume != "" ]; then
    args="--resume=$3 --load-last-model"
    aux="${@:4}"
else
    args=""
    aux="${@:3}"
fi

python main.py --learn --identifier=$identifier --game=$game $args $aux --algorithm=r2d2 \
--n-steps=5 --reward-shape --no-dropout --infinite-horizon --target=tde \
--batch=32 --seq-length=30 --burn-in=10 --seq-overlap=10 \
--hidden-features=256 --hiden-features-rnn=256 --clip=0 --discount=0.997 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.5 --epsilon-a=0.001 \
--epsilon-pre=0.00164 --epsilon-post=0.00164 --cpu-workers=24 \
--update-target-interval=1000 --n-tot=3125000 --checkpoint-interval=1000 \
--random-initialization=1000 --player-replay-size=2500 --update-memory-interval=50 \
--load-memory-interval=150 --replay-updates-interval=750 --replay-memory-size=2000000 \
