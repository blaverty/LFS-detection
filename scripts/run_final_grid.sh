#!/bin/bash

#PBS -l walltime=200:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=40g,vmem=40g
#PBS -e /hpf/largeprojects/davidm/blaverty/classify_lfs/parallel/final/logs/3e
#PBS -o /hpf/largeprojects/davidm/blaverty/classify_lfs/parallel/final/logs/3o
#PBS -m e

# qsub -t 1-30 bam2fastq.sh 
# qsub -t 1-100%10 bam2fastq.sh will run 10 jobs at a timemodule load python/3.8.1
# submits job for each task

# run prep.sh first

cd $PBS_O_WORKDIR

echo "Starting"

readarray -t files < /hpf/largeprojects/davidm/blaverty/classify_lfs/parallel/final/files
param=${files[$PBS_ARRAYID-1]}

qsub run_"$param".sh

echo EXIT STATUS $?
