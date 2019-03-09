#!/usr/bin/env bash

identifier=$1
game=$2
resume=$3
aux="${@:4}"

loc=`dirname "%0"`

if [ $resume != "new" ]; then
    resume="--resume=$3 --load-last-model"
    resume2="--resume=$3 --load-last-model"
    echo "Resume Experiment: $identifier $3"
else
    resume=""
    resume2="--resume=-1 --load-last-model"
    echo "New Experiment"
fi

args="--n-steps=3 --no-reward-shape --no-dropout --no-infinite-horizon \
--batch=512 --hidden-features=512 --clip=1 --discount=0.99 \
--termination-reward=0 --friction-reward=0 --priority-alpha=0.6 --priority-beta=0.4 --epsilon-a=0.001 \
--epsilon=0.00164 --cpu-workers=48 \
--update-target-interval=2500 --n-tot=3125000 --checkpoint-interval=5000 \
--random-initialization=1000 --player-replay-size=2500 --update-memory-interval=50 \
--load-memory-interval=50 --replay-updates-interval=1000 --replay-memory-size=2000000 --n-actors=16"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --learn --identifier=$identifier --game=$game $resume $args $aux --algorithm=rbi


if [ $? -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play $resume2 $args \
                            --identifier=$identifier --game=$game $aux \
                            --algorithm=rbi --play-episodes-interval=20 --max-frame=108000 &
    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play $resume2 $args \
                            --identifier=$identifier --game=$game $aux \
                            --algorithm=rbi --play-episodes-interval=20 --max-frame=108000 &
    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play $resume2 $args \
                            --identifier=$identifier --game=$game $aux \
                            --algorithm=rbi --play-episodes-interval=20 --max-frame=108000 &
    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play $resume2 $args \
                            --identifier=$identifier --game=$game $aux \
                            --algorithm=rbi --play-episodes-interval=20 --max-frame=108000 &
    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play $resume2 $args \
                            --identifier=$identifier --game=$game $aux \
                            --algorithm=rbi --play-episodes-interval=20 --max-frame=108000 &
    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play $resume2 $args \
                            --identifier=$identifier --game=$game $aux \
                            --algorithm=rbi --play-episodes-interval=20 --max-frame=108000 &

    CUDA_VISIBLE_DEVICES=0, python $loc/main.py --postprocess $resume2 $args \
                            --identifier=$identifier --game=$game $aux --algorithm=rbi &
fi
