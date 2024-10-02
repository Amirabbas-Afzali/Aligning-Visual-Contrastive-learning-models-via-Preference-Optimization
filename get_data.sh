#!/bin/bash

# See man sbatch or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=DOWNLOAD             # A nice readable name of your job, to see it in the queue
#SBATCH --output=/home/ali.rasekh/ambo/final/outputs/DOWNLOAD.out             # A nice readable name of your job, to see it in the queue
#SBATCH --error=/home/ali.rasekh/ambo/final/outputs/DOWNLOAD.err    
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=2           # Number of CPUs to request
#SBATCH --nodelist=gpunode102
#SBATCH --gpus=0                   # Number of GPUs to request

source /opt/conda/etc/profile.d/conda.sh
# Activate your environment, you have to create it first
conda activate RPO  #RVLM

# Your job script goes below this line
# /home/ali.rasekh/ambo/final/create_data.py
python3 /home/ali.rasekh/ambo/final/create_data.py
echo 'finished downloading'