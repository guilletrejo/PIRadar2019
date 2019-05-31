#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --ntask=4
#SBATCH --output=cnn_train-%j-salida.txt
#SBATCH --output=cnn_train-%j-error.txt

source activate pi_radar
sbatch python cnn_train.py
