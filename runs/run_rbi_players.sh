#!/usr/bin/env bash

identifier=$1
game=$2
resume=$3
aux="${@:4}"

loc=`dirname "%0"`

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

args="--algorithm=rbi --n-steps=3 --no-reward-shape --no-dropout --no-infinite-horizon \
--batch=512 --hidden-features=512 --clip=1 --discount=0.99 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.6 --priority-beta=0.4 --epsilon-a=0.001 \
--epsilon=0.00164 --cpu-workers=48 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=5000 \
--random-initialization=1000 --player-replay-size=2500 --update-memory-interval=50 \
--load-memory-interval=50 --replay-updates-interval=1000 --replay-memory-size=2000000 --n-actors=16"

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=0 $args $aux &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=1 $args $aux &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=2 $args $aux &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=3 $args $aux &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=4 $args $aux &
CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=5 $args $aux &

CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=6 $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=7 $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=8 $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=9 $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=10 $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=11 $args $aux &

CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=12 $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=13 $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=14 $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --actor-index=15 $args $aux &

CUDA_VISIBLE_DEVICES=3, python $loc/main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --evaluate --identifier=$identifier --resume=$resume --load-last-model --game=$game --play-episodes-interval=16 --wait=150 $tensor $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --evaluate --identifier=$identifier --resume=$resume --load-last-model --game=$game --play-episodes-interval=16 --wait=300 $tensor $args $aux &

