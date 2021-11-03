#!/bin/bash -l
#SBATCH --job-name=drqReach
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=gpu
# Request 2 gpus
#SBATCH --gres=gpu:1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=30

# specify the walltime e.g 20 mins
#SBATCH -t 20:20:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

source activate drqv2

# command to use
python3 train.py task=reach

# sbatch --partition=gpu train.sh
