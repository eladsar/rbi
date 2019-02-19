#!/bin/bash

nvidia-docker container run --rm -it --net=host --ipc=host -v $HOME/docker2:/workspace/data --name rbi1 local:rbi
