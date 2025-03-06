# Activation Extraction Package

This package provides tools for extracting, validating, and visualizing activations from transformer-based language models.

## Features

- Extract attention head activations, hidden states, and logits from transformer models
- Process single files or batches from directories
- Organize outputs by model and dataset
- Verify the integrity of extracted activations
- Visualize attention patterns, PCA projections, and token predictions

## Usage

### Command Line Interface

The simplest way to use this package is through the command line interface:

```bash
# From the root directory
python run_activations.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
                         --input_dir "outputs_small/gsm8k_test/raw_outputs" \
                         --dataset "gsm8k_test" \
                         --num_files 5

# Alternatively
python -m activation_extraction --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
                               --input_dir "outputs_small/gsm8k_test/raw_outputs" \
                               --dataset "gsm8k_test" \
                               --num_files 5
```

### Key Command Line Options

- `--model`: Name of the HuggingFace model to use
- `--input_dir`: Directory containing raw output files
- `--dataset`: Dataset name (used for organizing outputs)
- `--num_files`: Number of files to process (default: process all files)
- `--skip_extraction`: Skip extraction phase and only run verification
- `--skip_verification`: Skip verification phase and only run extraction
- `--verbose`: Print detailed information during processing

### Visualization Options

- `--visualize`: Run visualization on processed files
- `--viz_files`: Number of files to visualize (default: 1)
- `--viz_layers`: Comma-separated list of layers to visualize attention for (e.g., "0,10,20")
- `--viz_heads`: Comma-separated list of attention heads to visualize (e.g., "0,4")
- `--viz_modes`: Visualization modes to run (choices: "attention", "pca", "logits", "all")

### Programmatic Usage

You can also import the package functions directly in your Python code:

```python
import os
import sys

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from the activation_extraction package
from activation_extraction import (
    load_activation_file,
    visualize_attention_patterns,
    check_file_integrity
)

# Example: Load and analyze an activation file
file_path = "activations/DeepSeek-R1-Distill-Qwen-1.5B/gsm8k_test/output_0_activations.pt"
data = load_activation_file(file_path)

# Visualize attention patterns
visualize_attention_patterns(data, layer_idx=0, head_idx=0)
```

See the `use_activations.py` script for a complete example.

## Output Structure

Activations are organized in a model-centric directory structure:

```
activations/
  └── model_name/
      └── dataset_name/
          └── output_X_activations.pt

viz_output/
  └── model_name/
      └── dataset_name/
          └── output_X_activations_*_visualization.png
```

This makes it easy to compare outputs across different models and datasets.

## Development

To add new visualization types or processing methods, extend the appropriate files:

- `get_activations.py`: Core extraction logic
- `check_activations.py`: Validation functions
- `visualize_activations.py`: Visualization tools
- `run_batch_and_verify.py`: Orchestration and batch processing 