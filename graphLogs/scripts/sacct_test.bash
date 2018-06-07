#!/usr/bin/bash

sacct -a --format=Partition,Start,Submit > test.txt

python graphLogs.py test.txt



