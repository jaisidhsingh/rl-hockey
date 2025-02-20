SHELL := C:/Program Files/Git/bin/bash.exe

up-sac:
	scp -r "sac" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-vanilla-sac:
	scp -r "vanilla-sac" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-td3:
	scp -r "td3" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-utils:
	scp -r "utils" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-self-play-sac:
	scp -r "agents/self_play_pool" "stud369@login2.tcml.uni-tuebingen.de:~/agents"
	scp -r "agents/sac_agent.pt" "stud369@login2.tcml.uni-tuebingen.de:~/agents"
	scp -r "sac_self_play.py" "stud369@login2.tcml.uni-tuebingen.de:~"
	scp -r "sac_self_play.sbatch" "stud369@login2.tcml.uni-tuebingen.de:~"

fetch:
	scp -r "stud369@login2.tcml.uni-tuebingen.de:~/agents" .

local-self-play:
	python sac_self_play.py --initial_checkpoint agents/sac_agent.pt --n_step_td 3 --prioritized_replay

# login:
# 	ssh stud369@login2.tcml.uni-tuebingen.de

# start:
#	sbatch projects/sac/sac.sbatch

# build image:
#	singularity build --fakeroot container.sif container.def