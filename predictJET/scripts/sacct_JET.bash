#!/usr/bin/bash

module load AI/anaconda2-5.1.0_gpu

source activate $AI_ENVi

sacct -a -S 01/01/18 -X --format=UID,TimeLimit,Partition,Elapsed > ../logs/sacct_from_20180101

python predictJET.py ../logs/sacct_from_20180101



