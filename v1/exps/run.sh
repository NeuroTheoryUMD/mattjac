#!/bin/bash
# This script is used to run the experiment and restart it until it completes

# print a help line if no args are provided
if [ $# -lt 2 ]
then
    echo "Insufficient arguments supplied"
    echo "Usage: ./run.sh <experiment.py> <experiment_name> [--overwrite]"
    exit 1
fi

# if --overwrite flag is set, remove the existing experiment directory
if [[ $* == *--overwrite* ]]
then
    echo "Overwriting the previous experiment: $2"
    rm -rf "../experiments/$2"
fi

# loop until the file 'finished' is created
while [ ! -f "../experiments/$2/finished" ]
do
    # run the experiment
    python $1
done
