#!/bin/bash

# Configuration
SCRIPT_NAME="predict_triplets.py"
NUM_PROCESSES=12

echo "Launching $NUM_PROCESSES parallel processes..."

for i in $(seq 0 $((NUM_PROCESSES-1)))
do
    echo "Starting Shard $i"
    # Run python script with shard index and total count
    # Output is saved to shard_X.log
    python3 $SCRIPT_NAME $i $NUM_PROCESSES > "shard_$i.log" 2>&1 &
done

echo "Processes are running in the background."
echo "Use 'tail -f shard_0.log' to monitor progress."
wait
echo "All shards complete!"
