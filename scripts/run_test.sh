#!/bin/bash

#PBS -l walltime=200:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=50g,vmem=50g
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/test
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/test
#PBS -m e

module load python #3.9.7
cd $PBS_O_WORKDIR
echo "Starting"

python test.py 

echo EXIT STATUS $?
