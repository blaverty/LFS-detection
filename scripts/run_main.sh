#!/bin/bash

#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=5
#PBS -l mem=30g,vmem=30g
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/scripts/e_main
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/scripts/o_main
#PBS -m e

module load python/3.8.1

cd $PBS_O_WORKDIR
echo "Starting"

python main.py

echo EXIT STATUS $?
