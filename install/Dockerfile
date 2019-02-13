FROM pytorch/pytorch:latest

LABEL maintainer="Roth Ariel <rothar1@biu.ac.il>"
LABEL description="RBI code"


#Install wget:
#Install Tmux:
RUN  apt-get update && \
  apt-get install -y wget && \
  apt install -y tmux && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get install htop
RUN apt-get install iputils-ping
RUN apt install libgl1-mesa-glx

RUN rm /bin/sh
RUN ln -s /bin/bash /bin/sh

RUN mkdir -p /workspace/data
RUN mkdir -p /workspace/data/rbi
RUN mkdir -p /workspace/data/rbi/results
RUN mkdir -p /workspace/data/rbi/logs

RUN	mkdir -p /workspace/projects && \
	cd /workspace/projects && \
    git clone https://github.com/eladsar/rbi.git 
RUN cd /workspace/projects/rbi

RUN conda env create --file /workspace/projects/rbi/install/environment.yml
RUN /opt/conda/envs/torch1/bin/pip install atari-py

#RUN conda list

# Run app.py when the container launches
#CMD ["conda", "list"]

#CMD [ "source activate torch1 && exec python application.py" ]

# source activate torch1 && sh run_rbi_learner.sh debug qbert 0
