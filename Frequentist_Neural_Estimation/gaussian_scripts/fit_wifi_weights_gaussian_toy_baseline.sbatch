#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:1
#SBATCH -o outputs/fit_wifi_weights_gaussian_toy_baseline_%a.out
#SBATCH -e outputs/fit_wifi_weights_gaussian_toy_baseline_%a.err
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL

source activate YOUR_ENV_HERE
cd $SLURM_SUBMIT_DIR

python3 fit_wifi_weights_gaussian_toy_baseline.py ${SLURM_ARRAY_TASK_ID}
