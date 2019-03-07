#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` <algorithm> <identifier> <game> <resume> <aux>"
  exit 0
fi

algorithm=$1
all="${@:2}"

loc=`dirname "%0"`

case "$algorithm" in
    ("ape") bash $loc/runs/run_ape_learner.sh $all ;;
    ("rbi") bash $loc/runs/run_rbi_learner.sh $all ;;
    ("r2d2") bash $loc/runs/run_r2d2_learner.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac