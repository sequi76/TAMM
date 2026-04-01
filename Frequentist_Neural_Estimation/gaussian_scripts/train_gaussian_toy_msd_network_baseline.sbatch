#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1
#SBATCH -o outputs/train_gaussian_toy_msd_network_baseline_%a.out
#SBATCH -e outputs/train_gaussian_toy_msd_network_baseline_%a.err
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL

source activate YOUR_ENV_HERE
cd $SLURM_SUBMIT_DIR

python3 train_gaussian_toy_msd_network_baseline.py ${SLURM_ARRAY_TASK_ID}
