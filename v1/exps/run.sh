#!/bin/bash
# this script is used to run the experiment and restart it until it completes

# loop until the file 'finished' is created
while [ ! -f "../experiments/$1/finished" ]
do
    # run the experiment
    python $1.py
done
