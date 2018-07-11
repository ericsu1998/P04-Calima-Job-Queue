#!/usr/bin/bash

#Bash script for predicting job execution times
#1st positional argument ($1): file to store log data
#2nd positional arg ($2): file to store extracted data
#                         (if exists, merge, if doesn't, create new)

module load python3

msg1="Getting logs..."
echo $msg1

get_start_time=$SECONDS

squeue --start --format="%.18i %.19S" > $1

get_end_time=$SECONDS
get_time=$(($get_end_time-$get_start_time))
echo "Total of $get_time seconds elapsed for getting sacct logs"

msg2="Extracting data from logs..."
echo $msg2

extract_start_time=$SECONDS

python3 updatePredictedTimes.py $1 $2

extract_end_time=$SECONDS
extract_time=$(($extract_end_time-$extract_start_time))
echo "Total of $extract_time seconds elapsed for extracting data"






