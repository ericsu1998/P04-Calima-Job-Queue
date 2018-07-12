#!/usr/bin/bash

#Bash script for predicting job execution times
#1st positional argument ($1): file to store log data

extract_start_time=$SECONDS

msg1="Getting sacct logs..."
echo $msg1

sacct -a -P -S 01/01/18 -X --delimiter=',' --partition=RM --format=UID,TimeLimit,NNodes,Eligible,Start > $1

extract_end_time=$SECONDS
extract_time=$(($extract_end_time-$extract_start_time))
echo "Total of $extract_time seconds elapsed for getting sacct logs"

