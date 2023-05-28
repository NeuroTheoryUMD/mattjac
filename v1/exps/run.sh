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

# set the maximum number of iterations
max_iterations=100
sleep_time=10 # seconds

# initialize counter
counter=0

# loop until the file 'finished' is created or max_iterations is reached
while [[ ! -f "../experiments/$2/finished" ]] && [[ $counter -lt $max_iterations ]]
do
    # increment the counter
    ((counter++))
    
    # run the experiment
    python $1
    
    # sleep for 5 seconds
    sleep $sleep_time
done

if [[ $counter -eq $max_iterations ]]
then
    echo "Maximum number of iterations reached."
fi
