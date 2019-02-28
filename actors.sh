#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` <algorithm> <identifier> <game> <resume> <aux>"
  exit 0
fi

algorithm=$1
all="${@:2}"

loc=`dirname "%0"`

case "$algorithm" in
    ("ape") bash $loc/runs/run_ape_players.sh $all ;;
    ("rbi") bash $loc/runs/run_rbi_players.sh $all ;;
    ("r2d2") bash $loc/runs/run_r2d2_players.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac