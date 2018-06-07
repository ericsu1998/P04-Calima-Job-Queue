#!/usr/bin/bash

sacct -a -X --format=UID,TimeLimit,Partition,Elapsed > ../logs/sacct_20180607.txt

python predictJET.py ../logs/sacct_20180607.txt



