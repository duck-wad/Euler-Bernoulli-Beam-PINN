#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --account=def-rgracie
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
pip install --no-index --upgrade pip
pip install --no-index torch
pip install --no-index scikit-learn
pip install --no-index tqdm

python main.py 