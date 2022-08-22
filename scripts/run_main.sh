#!/bin/bash

#PBS -l walltime=100:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=40g,vmem=40g
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/scripts/logs/o
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/scripts/logs/e
#PBS -m e

module load python/3.8.1
cd $PBS_O_WORKDIR
echo "Starting"

python main.py 

echo EXIT STATUS $?
