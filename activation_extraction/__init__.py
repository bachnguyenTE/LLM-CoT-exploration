"""
Activation Extraction Package

This package contains utilities for extracting and analyzing activations from transformer models.
"""

from .get_activations import process_model_activations, process_directory, extract_attention_heads
from .check_activations import check_file_integrity, check_directory
from .visualize_activations import load_activation_file, visualize_attention_patterns, visualize_hidden_state_pca, visualize_top_logits

__all__ = [
    "process_model_activations",
    "process_directory",
    "extract_attention_heads",
    "check_file_integrity",
    "check_directory",
    "load_activation_file",
    "visualize_attention_patterns",
    "visualize_hidden_state_pca", 
    "visualize_top_logits"
] 