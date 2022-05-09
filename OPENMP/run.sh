# run.sh
# !/bin/sh
# PBS -N run
# PBS -l nodes=1
pssh -h $PBS_NODEFILE mkdir -p /home/s2120210448
pscp -h $PBS_NODEFILE /home/s2120210448/code/OPENMP/openmp /home/s2120210448
#mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2120210448/openmp
/home/s2120210448/code/OPENMP/openmp

