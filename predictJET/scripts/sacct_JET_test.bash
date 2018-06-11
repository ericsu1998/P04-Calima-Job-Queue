#!/usr/bin/bash

module load AI/anaconda2-5.1.0_gpu

source activate $AI_ENVi

sacct -a -S 01/01/18 -X --format=JobID,UID > ../logs/sacct_from_20180101_UID

sacct -a -S 01/01/18 -X --format=JobID,TimeLimit > ../logs/sacct_from_20180101_TimeLimit

sacct -a -S 01/01/18 -X --format=JobID,Partition > ../logs/sacct_from_20180101_Partition

sacct -a -S 01/01/18 -X --format=JobID,Elapsed > ../logs/sacct_from_20180101_Elapsed

python predictJET.py ../logs/sacct_from_20180101_UID ../logs/sacct_from_20180101_TimeLimit ../logs/sacct_from_20180101_Partition ../logs/sacct_from_20180101_Elapsed








