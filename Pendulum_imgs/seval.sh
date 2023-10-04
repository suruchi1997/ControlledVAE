#!/bin/bash
#SBATCH --job-name=cclm-peni-eval-%A_%a
#SBATCH --output=cclm-peni-eval-%A_%a.out
#SBATCH --error=cclm-peni-eval-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
module load anaconda/3.9
source /home/$USER/.bashrc
conda activate cclm-cuda
python eval.py --rs $1 --beta $2