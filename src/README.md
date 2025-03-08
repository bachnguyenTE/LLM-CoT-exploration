# LLM Chain-of-Thought Exploration - Code Structure

This document provides an overview of the reorganized codebase for exploring Large Language Model chain-of-thought reasoning.

## Directory Structure

```
src/
├── experiments/             # Experimental code for different datasets
│   ├── gsm8k/               # GSM8K math reasoning experiments
│   └── penguin/             # Penguin classification experiments
├── utils/                   # Utility functions used across experiments
├── models/                  # Model-related code and interfaces
├── analysis/                # Analysis scripts for experimental results
└── visualization/           # Visualization tools for results
```

## Experiments

### GSM8K Experiments
The GSM8K directory contains scripts for running experiments on the GSM8K math reasoning dataset:

- `GSM8K_generate_from_test.py`: Generates responses from models using the GSM8K test set
- `GSM8K_generate_from_train.py`: Generates responses from models using the GSM8K training set
- `anti_GSM8K_generate_from_train.py`: Anti-CoT experiments with GSM8K training data
- `unthink_GSM8K_generate_from_test.py`: Experiments without thinking steps on GSM8K test data
- `unthink_GSM8K_generate_from_train.py`: Experiments without thinking steps on GSM8K training data
- `intervention_GSM8K_generate_from_test.py`: Intervention experiments on GSM8K test data
- Various shell scripts for running these experiments with different parameters

### Penguin Experiments
The penguin directory contains scripts for penguin classification experiments:

- `penguin_generator.py`: Generates model responses for penguin classification
- `antipenguin_generator.py`: Anti-CoT experiments for penguin classification
- `super_antipenguin_generator.py`: Enhanced anti-CoT experiments
- `unthink_penguin_generator.py`: Experiments without thinking steps
- `intervention_penguin_generator.py`: Intervention experiments for penguin classification
- `run_penguin_analysis.py`: Analysis of penguin classification experiments
- `parse_penguin_results.py`: Parses results from penguin classification experiments
- Various shell scripts for running these experiments with different parameters

## Utils
Utility functions used across different experiments:

- `run_activations.py`: Script for extracting model activations
- `use_activations.py`: Script for using extracted activations
- `run_batch_and_verify.py`: Batch processing utilities

## Analysis
Scripts for analyzing experimental results:

- `run_analysis.py`: General analysis script
- `compare_answers_GSM8K.py`: Compares different model answers on GSM8K tasks
- `compare_intervention_GSM8K.py`: Analyzes intervention results on GSM8K tasks

## Models
Model-related code and interfaces (placeholder for future model-specific code).

## Visualization
Visualization tools for experimental results (placeholder for future visualization code).

## How to Run Experiments

To run an experiment, navigate to the relevant experiment directory and execute the corresponding script:

```bash
# Example: Running a GSM8K experiment
cd src/experiments/gsm8k
python GSM8K_generate_from_test.py --model "your-model-name" --output_dir "path/to/output"

# Example: Running a penguin experiment
cd src/experiments/penguin
python penguin_generator.py --start 0 --end 10 --temperature 0.6
```

Alternatively, you can use the provided shell scripts:

```bash
# Example: Running a GSM8K experiment using a shell script
cd src/experiments/gsm8k
./run_GSM8K_test.sh

# Example: Running a penguin experiment using a shell script
cd src/experiments/penguin
./run_penguins.sh
```

## File Paths in Scripts

Note that all scripts have been updated to use relative imports and file paths based on the new directory structure. If you encounter any path-related issues, please check the import statements and file paths in the script. 