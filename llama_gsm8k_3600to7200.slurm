#!/bin/bash
#SBATCH -p compute
#SBATCH -t 3-0
#SBATCH -C epyc
#SBATCH -c 128
#SBATCH --mem=400G

source anaconda3/etc/profile.d/conda.sh
conda activate transformer-env
cd projects/CoT-exploration
./run_anti_GSM8K_train.sh 3600 7200 12
