#!/bin/bash

#source params.sh

# nvidia-docker run --rm -it ${NET} ${IPC} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash
#docker container run --rm --name ${CONTNAME} ${IMAGENAME} 
#--shm-size=<requested memory size>
docker container run --rm -it --net=host --ipc=host --name rbi1 local:rbi bash
docker container run --rm -it --net=host --ipc=host --name test pytorch/pytorch:latest bash