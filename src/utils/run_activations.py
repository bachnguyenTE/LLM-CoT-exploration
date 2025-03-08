#!/usr/bin/env python
"""
Entry point for the activation extraction system.

This script serves as a simple entry point to run the batch and verify system
for extracting activations from transformer models.
"""

import sys
import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Ensure the activation_extraction directory is in the path
sys.path.insert(0, project_root)

# Import the main function from run_batch_and_verify
from activation_extraction.run_batch_and_verify import main

if __name__ == "__main__":
    # Run the main function, which will parse command-line arguments and execute
    main() 