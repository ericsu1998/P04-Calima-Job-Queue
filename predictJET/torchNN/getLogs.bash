#!/usr/bin/bash

#Bash script for predicting job execution times

module load AI/anaconda3-5.1.0_gpu

source activate $AI_ENV

extract_start_time=$SECONDS

msg1="Getting sacct logs..."
echo $msg1

sacct -a -P -S 01/01/18 -X --format=UID,Partition,TimeLimit,Elapsed > sacct_from_20180101.log

#msg2="Extracting done!"
#echo $msg2

#Internal use only: uncomment when done
#Takes 25 seconds to extract logs
extract_end_time=$SECONDS
extract_time=$(($extract_end_time-$extract_start_time))
echo "Total of $extract_time seconds elapsed for getting sacct logs"

#Uncomment for full pipeline
python3 torchNN.py sacct_from_20180101.log








