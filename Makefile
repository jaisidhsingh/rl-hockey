SHELL := C:/Program Files/Git/bin/bash.exe

us:
	scp -r "sac" "stud369@login2.tcml.uni-tuebingen.de:~/projects"

us2:
	scp -r "sac2" "stud369@login2.tcml.uni-tuebingen.de:~/projects"

ut:
	scp -r "td3" "stud369@login2.tcml.uni-tuebingen.de:~/projects"

fetch:
	scp -r "stud369@login2.tcml.uni-tuebingen.de:~/agents" .

#pw: peika7Tein

# login:
# 	ssh stud369@login2.tcml.uni-tuebingen.de

# start:
#	sbatch projects/sac/sac.sbatch

# build image:
#	singularity build --fakeroot container.sif container.def