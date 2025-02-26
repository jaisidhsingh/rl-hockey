SHELL := C:/Program Files/Git/bin/bash.exe

up-sac:
	scp -r "sac" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-td3:
	scp -r "td3" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-utils:
	scp -r "utils" "stud369@login2.tcml.uni-tuebingen.de:~/"

up-self-play-sac:
	scp -r "agents/sac_self_play_pool5" "stud369@login2.tcml.uni-tuebingen.de:~/agents"
	scp -r "agents/sac_agent_sp1M.pt" "stud369@login2.tcml.uni-tuebingen.de:~/agents"
	scp -r "sac_self_play.py" "stud369@login2.tcml.uni-tuebingen.de:~"
	scp -r "sac_self_play.sbatch" "stud369@login2.tcml.uni-tuebingen.de:~"

up-self-play-td3:
	scp -r "agents/td3_self_play_pool2_2" "stud369@login2.tcml.uni-tuebingen.de:~/agents"
	scp -r "agents/td3_agent_sp1M.pt" "stud369@login2.tcml.uni-tuebingen.de:~/agents"
	scp -r "td3_self_play.py" "stud369@login2.tcml.uni-tuebingen.de:~"
	scp -r "td3_self_play.sbatch" "stud369@login2.tcml.uni-tuebingen.de:~"

fetch:
	scp -r "stud369@login2.tcml.uni-tuebingen.de:~/agents" tcml-agents

fetch-self-play-sac:
	scp -r "stud369@login2.tcml.uni-tuebingen.de:~/agents/sac_self_play_pool5" agents/

fetch-self-play-td3:
	scp -r "stud369@login2.tcml.uni-tuebingen.de:~/agents/td3_self_play_pool2_3" agents/t

local-self-play-sac:
	python sac_self_play.py --n_step_td 3 --prioritized_replay --checkpoint_dir agents/sac_self_play_pool4_1 --initial_checkpoint agents/sac_agent_sp1M.pt --save_opponent_interval 50000

local-self-play-td3:
	python td3_self_play.py --checkpoint_dir agents/td3_self_play_pool2 --initial_checkpoint agents/td3_agent_sp1M.pt --save_opponent_interval 50000

# login:
# 	ssh stud369@login2.tcml.uni-tuebingen.de

# start:
#	sbatch projects/sac/sac.sbatch

# build image:
#	singularity build --fakeroot container.sif container.def

# comp-client:
# 	python run_client.py --server-url comprl.cs.uni-tuebingen.de --server-port 65335 \
# 		--token b4d343c8-aa2d-4359-8a90-bb41a097fb77 \
# 		--args --agent=sac_sp --checkpoint_path=agents/sac_agent_strongest.pt

local-comp-client:
	bash autorestart.sh --server-url comprl.cs.uni-tuebingen.de --server-port 65335 \
		--token af896257-7d1e-48fe-9606-5dd76a170965 \
		--args --agent=sac_sp --checkpoint-path="agents/sac_agent_strongest.pt"
	
	bash autorestart.sh --server-url comprl.cs.uni-tuebingen.de --server-port 65335 \
		--token f959d4bb-1dd9-4297-bcdc-ef3c724133e5 \
		--args --agent=td3_sp --checkpoint-path="agents/td3_agent_strongest.pt"

up-client:
	scp -r "run_client.py" "stud369@login2.tcml.uni-tuebingen.de:~"
	scp -r "autorestart.sh" "stud369@login2.tcml.uni-tuebingen.de:~"

comp-client:
	sed -i 's/\r$//' autorestart.sh

	singularity run containers/container-client.sif bash autorestart.sh --server-url comprl.cs.uni-tuebingen.de --server-port 65335 --token af896257-7d1e-48fe-9606-5dd76a170965 --args --agent=sac_sp --checkpoint-path="agents/sac_agent_strongest.pt"

	singularity run containers/container-client.sif bash autorestart.sh --server-url comprl.cs.uni-tuebingen.de --server-port 65335 --token f959d4bb-1dd9-4297-bcdc-ef3c724133e5 --args --agent=td3_sp --checkpoint-path="agents/td3_agent_strongest.pt"