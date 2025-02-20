SHELL := C:/Program Files/Git/bin/bash.exe

up-sac:
	scp -r "sac" "stud369@login2.tcml.uni-tuebingen.de:~/projects"

up-td3:
	scp -r "td3" "stud369@login2.tcml.uni-tuebingen.de:~/projects"

fetch:
	scp -r "stud369@login2.tcml.uni-tuebingen.de:~/agents" .

# login:
# 	ssh stud369@login2.tcml.uni-tuebingen.de

# start:
#	sbatch projects/sac/sac.sbatch

# build image:
#	singularity build --fakeroot container.sif container.def