#!/bin/bash

#SBATCH -t 200:00:00
#SBATCH -N 1 -c 20
#SBATCH --mem=50g
#SBATCH -o /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/gbt_10_o
#SBATCH -e /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/gbt_10_e

# umap needs a lot more memory

module load python #/3.9.2
echo "Starting"

python /hpf/largeprojects/davidm/blaverty/classify_lfs/scripts/main.py 

echo EXIT STATUS $?
