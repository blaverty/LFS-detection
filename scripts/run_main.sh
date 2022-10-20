#!/bin/bash

#SBATCH -t 100:00:00
#SBATCH -N 1 -c 40
#SBATCH --mem=30g
#SBATCH -o /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/log_o_classif
#SBATCH -e /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/log_e_classif

# umap needs a lot more memory

module load python #/3.9.2
cd $PBS_O_WORKDIR
echo "Starting"

python main.py 

echo EXIT STATUS $?
