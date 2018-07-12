#!/usr/bin/bash

#Runs queueWaitTimeNN script
#$1 Log file as data

module load AI/anaconda3-5.1.0_gpu

source activate $AI_ENV

python3 queueWaitTimeNN.py $1







