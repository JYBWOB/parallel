# run.sh
# !/bin/sh
# PBS -N test
# PBS -l nodes=4
pssh -h $PBS_NODEFILE mkdir -p /home/s2120210448
pscp -h $PBS_NODEFILE /home/s2120210448/openmp /home/s2120210448
#mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2120210448/openmp
./openmp

