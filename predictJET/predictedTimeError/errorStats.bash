#!/usr/bin/bash

#Bash script for predicting job execution times
#1st positional arg ($1): file to get dictionary of predicted start times

module load AI/anaconda3-5.1.0_gpu
source activate $AI_ENV

msg1="Calculating error..."
echo $msg1

get_start_time=$SECONDS

python3 errorStats.py $1

get_end_time=$SECONDS
get_time=$(($get_end_time-$get_start_time))
echo "Total of $get_time seconds elapsed for calculating error stats"






