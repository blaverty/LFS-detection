#!/bin/bash

#PBS -l walltime=200:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=250g,vmem=250g 
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/log_o_none_option
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/log_e_none_option
#PBS -m e

# umap needs a lot more memory

module load python #/3.9.2
cd $PBS_O_WORKDIR
echo "Starting"

python main.py 

echo EXIT STATUS $?
