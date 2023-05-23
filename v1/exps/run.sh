#!/bin/bash
# this script is used to run the experiment and restart it until it completes

# print a help line if no args are provided
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Usage: ./run.sh <experiment.py> <experiment_name>"
    exit 1
fi

# loop until the file 'finished' is created
while [ ! -f "../experiments/$2/finished" ]
do
    # run the experiment
    python $1
done
