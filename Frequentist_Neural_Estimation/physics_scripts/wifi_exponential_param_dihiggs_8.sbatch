#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 0-02:00
#SBATCH --gres=gpu:1
#SBATCH -o outputs/wifi_exponential_param_dihiggs_8_%a.out
#SBATCH -e outputs/wifi_exponential_param_dihiggs_8_%a.err
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL

source activate YOUR_ENV_HERE
cd $SLURM_SUBMIT_DIR

python3 wifi_exponential_param_dihiggs_8.py ${SLURM_ARRAY_TASK_ID} --lam ${LAM:-1.0}
