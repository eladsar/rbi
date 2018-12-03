#!/usr/bin/env bash

identifier=$1
resume=$2
game=$3


python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=0 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=1 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=2 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=3 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=4 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=5 &

python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=6 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=7 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=8 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=9 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=10 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=11 &

python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --n-actors=22 --actor-index=12 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --n-actors=22 --actor-index=13 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --n-actors=22 --actor-index=14 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --n-actors=22 --actor-index=15 &

python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=16 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=17 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --n-actors=22 --actor-index=18 &

python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=1 --n-actors=22 --actor-index=19 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=2 --n-actors=22 --actor-index=20 &
python main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --n-actors=22 --actor-index=21 &

python main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game &
python main.py --choose --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --play-episodes-interval=24 --wait=150 &
python main.py --choose --identifier=$identifier --resume=$resume --load-last-model --game=$game --cuda-default=3 --play-episodes-interval=24 --wait=300 &

