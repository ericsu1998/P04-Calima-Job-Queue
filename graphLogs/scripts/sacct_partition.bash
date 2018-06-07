#!/usr/bin/bash

sacct -S 01/01/18 -a --format=Partition,Start,Submit > ../logs/sacct_partition_20180101_20180607.txt

python graphLogs.py ../logs/sacct_partition_20180101_20180607.txt



