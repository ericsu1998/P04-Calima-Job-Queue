#!/usr/bin/bash

module load AI/anaconda2-5.1.0_gpu

source activate $AI_ENV

sacct -a -S 01/01/18 -X --format=TimeLimit,Elapsed > ../logs/sacct_time_from_20180101

python predictJET_time.py ../logs/sacct_time_from_20180101



