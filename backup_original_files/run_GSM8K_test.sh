#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <start> <end> <num_subdivisions> [model_name] [output_folder]"
    echo "Example: $0 0 1000 10 \"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\" \"outputs_small\""
    exit 1
fi

start=$1
end=$2
num_subdivisions=$3

# Set default values for optional arguments
model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
output_folder="outputs"

# Override with provided arguments if they exist
if [ "$#" -ge 4 ]; then
    model_name=$4
fi

if [ "$#" -ge 5 ]; then
    output_folder=$5
fi

echo "Using model: $model_name"
echo "Output folder: $output_folder"

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
    
    # Run the command in background
    python GSM8K_generate_from_test.py --start $subdivision_start --end $subdivision_end --model_name "$model_name" --output_folder "$output_folder" &
    
    # Wait for 60 seconds before launching the next process, except for the last one
    if [ $((i + 1)) -lt $num_subdivisions ]; then
        echo "Waiting 60 seconds before launching next process..."
        sleep 60
    fi
done

# Wait for all processes to complete
echo "Waiting for all processes to complete..."
wait
echo "All processes completed!" 