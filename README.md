# rbi
Implementation of distributed RL algorithms:

Baselines:
1. Ape-X (https://arxiv.org/abs/1803.00933)
2. R2D2 (https://openreview.net/pdf?id=r1lyTjAqYX)
3. PPO (https://arxiv.org/abs/1707.06347)

RBI:
A safe reinforcement learning algorithm 

Currently, supported environment is ALE
## How to run

A distributed RL agent is composed of a single learning process and multiple actor process.
Therefore, we need to execute two bash scripts one for the learner and one for the multiple actors.

chose \<algorithm> as one of rbi|ape|ppo|r2d2|rbi_rnn 

### Run Learner:

sh run_\<algorithm>_learner.sh \<identifier> \<game> \<new|resume>

resume is a number of experiment to resume.
For example:

sh run_rbi_players.sh qbert_debug qbert new

starts a new experiment, while:

sh run_rbi_players.sh qbert_debug qbert 3

resumes experiment 3 with identifier qbert_debug

### Run Actors:

sh run_\<algorithm>_player.sh \<identifier> \<game> \<resume>

### Run Evaluation player:

right now there are two evaluation players in each actors script

### Terminate a live run:

1. ctrl-c from the learner process terminal
2. pkill -f "main.py"  (kills all the live actor processes)
3. rm -r /dev/shm/<your name>/rbi/* (clear the ramdisk filesystem)

### Setup prerequisites before running the code

To login: 
ssh \<username>@\<server-address>

Use ssh-keygen and ssh-copy-id to avoid passwords:
ssh-keygen
ssh-copy-id -i ~/.ssh/id_rsa user@host

Install Anaconda:
copy anaconda file to server and run:
sh Anaconda3-2018.12-Linux-x86_64.sh

Install Tmux:
make new directory called tmux_tmp
copy ncurses.tar.gz and tmx-2.5.tar.gz to tmux_tmp directory
copy install_tmux.sh to server and run
./install_tmux.sh

mkdir -p ~/data/rbi/results

mkdir -p ~/data/rbi/logs

mkdir -p ~/projects

cd ~/projects

git clone https://github.com/eladsar/rbi.git

cd ~/projects/rbi

conda env create -f environment.yml

source activate torch1

pip install atari-py



