#!/usr/bin/bash

#Bash script for predicting job execution times

module load AI/anaconda3-5.1.0_gpu

source activate $AI_ENV

python3 torchNN.py sacct_from_20180101.log








