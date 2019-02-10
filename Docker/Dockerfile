FROM pytorch/pytorch:latest

LABEL maintainer="Roth Ariel <rothar1@biu.ac.il>"
LABEL description="RBI code"


#Install wget:
#Install Tmux:
RUN  apt-get update && \
  apt-get install -y wget && \
  apt install -y tmux && \
  rm -rf /var/lib/apt/lists/*


#Install Tmux:
#RUN mkdir -p /workspace/tmux_tmp
#ADD files/ncurses.tar.gz /workspace/tmux_tmp/
#ADD files/install_tmux.sh /workspace
##ELAD error
#RUN ./install_tmux.sh



RUN mkdir -p /workspace/data
RUN mkdir -p /workspace/data/rbi
RUN mkdir -p /workspace/data/rbi/results
RUN mkdir -p /workspace/data/rbi/logs

#RUN mkdir /workspace/rbi_atari
#RUN mkdir /workspace/rbi_atari/results
#RUN mkdir /workspace/rbi_atari/logs

#RUN  /opt/conda/bin/conda install -c anaconda pandas && \
#	 /opt/conda/bin/conda install -c conda-forge matplotlib && \
#	 /opt/conda/bin/conda install -c menpo opencv && \
#	 /opt/conda/bin/conda install -c conda-forge tensorboardx && \
#     /opt/conda/bin/conda clean -ya

#RUN pip install comet_ml && \
#	pip install parse && \
#	pip install atari_py


RUN	mkdir -p /workspace/projects && \
	cd /workspace/projects && \
    git clone https://github.com/eladsar/rbi.git 
RUN cd /workspace/projects/rbi

RUN conda env create --file /workspace/projects/rbi/environment.yml
#RUN source activate torch1
#RUN pip install atari-py
#RUN conda list

# Run app.py when the container launches
#CMD ["conda", "list"]

#CMD [ "source activate torch1 && exec python application.py" ]

# source activate torch1 && sh run_rbi_learner.sh debug qbert 0
