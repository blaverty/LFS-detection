#!/bin/bash

#PBS -l walltime=200:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=40g,vmem=40g
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/log_o_newfeat_noumap_lessmem_50
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/logs/log_e_newfeat_noumap_lessmem_50
#PBS -m e

# umap needs a lot more memory

module load python #/3.9.2
cd $PBS_O_WORKDIR
echo "Starting"

python /hpf/largeprojects/davidm/blaverty/classify_lfs/scripts/main.py 

echo EXIT STATUS $?
