#!/bin/bash
#PBS -l ncpus=16,ngpus=1
#PBS -l mem=120GB
#PBS -l jobfs=200GB
#PBS -q dgxa100
#PBS -P wa66
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/wa66+scratch/wa66
#PBS -l wd

source /g/data/wa66/Xiangyu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

python3 /home/561/xz4320/Code_Switch/asru/code_switch_withhint.py &> /home/561/xz4320/Code_Switch/asru/asruhint.log