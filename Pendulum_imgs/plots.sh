#!/bin/bash
#SBATCH --job-name=cclm-pen-img-%A_%a
#SBATCH --output=cclm-pen-img-%A_%a.out
#SBATCH --error=cclm-pen-img-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --nodes=1
module load anaconda/3.9
source /home/$USER/.bashrc
conda activate cclm-cuda
python ../csv_com.py
python ../plots.py