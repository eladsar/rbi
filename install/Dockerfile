FROM pytorch/pytorch:latest

LABEL maintainer="Roth Ariel <rothar1@biu.ac.il>"
LABEL description="RBI code"


#Install wget, tmux, libgl1-mesa-glx 
RUN  apt-get update && \
  apt-get install -y wget && \
  apt install -y tmux && \
  apt-get install -y htop && \
  apt-get install -y iputils-ping && \
  apt install -y libgl1-mesa-glx 
RUN apt-get install -y openssh-server
RUN  rm -rf /var/lib/apt/lists/*


RUN ssh-keygen -A
ADD ./sshd_config /etc/ssh/sshd_config
RUN echo 'root:aimlab' | chpasswd




#RUN rm /bin/sh
#RUN ln -s /bin/bash /bin/sh

RUN mkdir -p /workspace/data
RUN mkdir -p /workspace/data/rbi
RUN mkdir -p /workspace/data/rbi/results
RUN mkdir -p /workspace/data/rbi/logs


RUN	mkdir -p /workspace/projects && \
	cd /workspace/projects && \
    git clone https://github.com/eladsar/rbi.git 
RUN cd /workspace/projects/rbi

RUN conda update -n base -c defaults conda
RUN conda env create --file /workspace/projects/rbi/install/environment.yml
RUN /opt/conda/envs/torch1/bin/pip install atari-py

CMD /usr/sbin/sshd -D

# source activate torch1 && sh run_rbi_learner.sh debug qbert 0
