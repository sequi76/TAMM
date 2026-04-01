#!/bin/bash
#SBATCH -n 1
#SBATCH -p iaifi_gpu
#SBATCH -t 0-01:00
#SBATCH --gres=gpu:1
#SBATCH -o outputs/wifi_baseline_gaussian_toy_%a.out
#SBATCH -e outputs/wifi_baseline_gaussian_toy_%a.err
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL

source activate YOUR_ENV_HERE
cd $SLURM_SUBMIT_DIR

python3 wifi_baseline_gaussian_toy.py ${SLURM_ARRAY_TASK_ID}
