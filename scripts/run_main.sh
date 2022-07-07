#!/bin/bash

#PBS -l walltime=100:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=30g,vmem=30g
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/all_log_o5
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/all_log_e5
#PBS -m e

module load python/3.8.1
cd $PBS_O_WORKDIR
echo "Starting"

python main.py 

echo EXIT STATUS $?
