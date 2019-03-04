#!/bin/bash

nvidia-docker container run --rm -it --net=host --ipc=host \
	--mount type=bind,source="$(pwd)"/data_docker,target=/workspace/data \
	--name rbi1 \
	-p 2201:22 \
	local:rbi
