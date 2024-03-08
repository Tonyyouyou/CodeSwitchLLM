#!/bin/bash
#PBS -l ncpus=16,ngpus=1
#PBS -l mem=120GB
#PBS -l jobfs=200GB
#PBS -q dgxa100
#PBS -P wa66
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/wa66+scratch/wa66
#PBS -l wd

export HF_HOME="/g/data/wa66/Xiangyu/cache/huggingface"
module load python3/3.9.2
module load openmpi/4.0.2
module load hdf5/1.10.5
source /home/561/xz4320/enviroment/pytorch_3.9.2/bin/activate

python3 /home/561/xz4320/Code_Switch/seame/code_switch_nohint_chinese.py &> ./chnohint.log 