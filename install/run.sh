#!/bin/bash

#source params.sh

# nvidia-install run --rm -it ${NET} ${IPC} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash
#install container run --rm --name ${CONTNAME} ${IMAGENAME}
#--shm-size=<requested memory size>
docker container run --rm -it --net=host --ipc=host --name rbi1 local:rbi bash

nvidia-docker run --rm -it --net=host --ipc=host --name rbi1 local:rbi bash
