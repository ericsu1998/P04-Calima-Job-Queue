#!/usr/bin/bash

#Bash script for predicting job execution times
#1st positional argument ($1): file to store log data (start times of CG jobs)
#2nd positional arg ($2): file to get dictionary of predicted start times

module load AI/anaconda3-5.1.0_gpu
source activate $AI_ENV

msg1="Getting logs..."
echo $msg1

get_start_time=$SECONDS

sacct -P -X -a --delimiter="," --format=Start,JobID > $1

get_end_time=$SECONDS
get_time=$(($get_end_time-$get_start_time))
echo "Total of $get_time seconds elapsed for getting start times of CG jobs"

msg2="Calculating difference between predicted and actual start times..."
echo $msg2

get_error_start_time=$SECONDS

python3 updateActualTimes.py $1 $2

get_error_end_time=$SECONDS
get_error_time=$(($get_error_end_time-$get_error_start_time))
echo "Total of $get_error_time seconds elapsed for calculating error"






