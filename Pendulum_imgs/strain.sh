#!/bin/bash
#SBATCH --job-name=cclm-peni-train-%A_%a
#SBATCH --output=cclm-peni-train-%A_%a.out
#SBATCH --error=cclm-peni-train-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --nodes=1
module load anaconda/3.9
source /home/$USER/.bashrc
conda activate cclm-cuda
python train.py --rs $1 --beta $2