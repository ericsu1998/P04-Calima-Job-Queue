#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH -p GPU-shared

./torchNN.bash
