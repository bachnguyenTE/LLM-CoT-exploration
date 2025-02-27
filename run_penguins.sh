#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start> <end> <num_subdivisions> <temperature>"
    echo "Example: $0 0 40 10 0.6"
    exit 1
fi

start=$1
end=$2
num_subdivisions=$3
temperature=$4

# Validate temperature is a number between 0.1 and 1.0
if ! [[ $temperature =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$temperature < 0.1" | bc -l) )) || (( $(echo "$temperature > 1.0" | bc -l) )); then
    echo "Error: temperature must be a number between 0.1 and 1.0"
    exit 1
fi

# Validate num_subdivisions is a positive integer
if ! [[ "$num_subdivisions" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_subdivisions must be a positive integer"
    exit 1
fi

# Calculate size of each subdivision
range=$((end - start))
subdivision_size=$(( (range + num_subdivisions - 1) / num_subdivisions ))

# Launch processes for each subdivision
for ((i=0; i<num_subdivisions; i++)); do
    subdivision_start=$((start + i * subdivision_size))
    subdivision_end=$((subdivision_start + subdivision_size))
    
    # Ensure we don't exceed the overall end value
    if [ $subdivision_end -gt $end ]; then
        subdivision_end=$end
    fi
    
    # Skip if subdivision_start has reached or exceeded end
    if [ $subdivision_start -ge $end ]; then
        continue
    fi
    
    echo "Starting process $((i+1)) of $num_subdivisions (range: $subdivision_start to $subdivision_end)"
    python penguin_generator.py --start $subdivision_start --end $subdivision_end --temperature $temperature &
    
    # Wait for 60 seconds before launching the next process, except for the last one
    if [ $((i + 1)) -lt $num_subdivisions ]; then
        echo "Waiting 60 seconds before launching next process..."
        sleep 60
    fi
done

wait