#!/usr/bin/env python
"""
Example of how to import and use the activation extraction package directly.

This script demonstrates how to use the activation extraction functions 
directly in your code.
"""

import os
import sys
from pathlib import Path

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from the activation_extraction package
from activation_extraction import (
    load_activation_file,
    visualize_attention_patterns,
    check_file_integrity
)

def list_activation_files(base_dir="activations", model_name=None, dataset_name=None):
    """List all activation files in the given directory structure."""
    if model_name and dataset_name:
        search_dir = os.path.join(base_dir, model_name, dataset_name)
    elif model_name:
        search_dir = os.path.join(base_dir, model_name)
    else:
        search_dir = base_dir
    
    print(f"Searching for activation files in: {search_dir}")
    
    # Find all activation files
    activation_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith("_activations.pt"):
                activation_files.append(os.path.join(root, file))
    
    return activation_files

def analyze_activation_file(file_path):
    """Load and analyze a single activation file."""
    print(f"\nAnalyzing file: {file_path}")
    
    # First check the file integrity
    is_valid = check_file_integrity(file_path, verbose=True)
    
    if not is_valid:
        print("File integrity check failed. Skipping analysis.")
        return
    
    # Load the file
    data = load_activation_file(file_path)
    if data is None:
        print("Failed to load activation file.")
        return
    
    # Print some basic statistics
    attention_layers = len(data["attention_activations"])
    num_heads = data["attention_activations"][0].shape[2] if 0 in data["attention_activations"] else 0
    seq_length = data["input_ids"].shape[1]
    hidden_dim = data["final_hidden_state"].shape[2]
    
    print(f"Model: {data['model_name']}")
    print(f"Sequence length: {seq_length}")
    print(f"Number of attention layers: {attention_layers}")
    print(f"Number of attention heads per layer: {num_heads}")
    print(f"Hidden state dimension: {hidden_dim}")
    
    return data

def main():
    # Example: list and analyze activation files
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    dataset_name = "gsm8k_test"
    
    activation_files = list_activation_files(
        base_dir="activations",
        model_name=model_name,
        dataset_name=dataset_name
    )
    
    print(f"Found {len(activation_files)} activation files")
    
    # Analyze the first file (if any)
    if activation_files:
        first_file = activation_files[0]
        analyze_activation_file(first_file)
    else:
        print("No activation files found.")

if __name__ == "__main__":
    main() 