#!/usr/bin/env python3
"""
Script to list all files in the codebase with their descriptions.
Useful for getting an overview of the project structure.
"""

import os
from pathlib import Path
import json

# Mapping of file extensions to descriptions
EXTENSION_DESCRIPTIONS = {
    ".py": "Python source code",
    ".sh": "Shell script",
    ".md": "Markdown documentation",
    ".json": "JSON data file",
    ".ipynb": "Jupyter notebook",
    ".txt": "Text file",
}

# Mapping of specific files to descriptions
FILE_DESCRIPTIONS = {
    "penguin_generator.py": "Generates model responses for the penguin classification task",
    "antipenguin_generator.py": "Generates anti-CoT responses for the penguin classification task",
    "super_antipenguin_generator.py": "Generates enhanced anti-CoT responses for penguin task",
    "unthink_penguin_generator.py": "Generates responses without thinking steps for penguin task",
    "intervention_penguin_generator.py": "Generates responses with interventions for penguin task",
    "run_penguin_analysis.py": "Analyzes results from penguin classification experiments",
    "parse_penguin_results.py": "Parses model outputs from penguin classification",
    "GSM8K_generate_from_test.py": "Generates responses from GSM8K test set",
    "GSM8K_generate_from_train.py": "Generates responses from GSM8K training set",
    "anti_GSM8K_generate_from_train.py": "Generates anti-CoT responses from GSM8K training set",
    "unthink_GSM8K_generate_from_test.py": "Generates responses without thinking steps from GSM8K test set",
    "unthink_GSM8K_generate_from_train.py": "Generates responses without thinking steps from GSM8K training set",
    "intervention_GSM8K_generate_from_test.py": "Generates responses with interventions from GSM8K test set",
    "run_analysis.py": "General analysis script for experimental results",
    "compare_answers_GSM8K.py": "Compares different model answers on GSM8K tasks",
    "compare_intervention_GSM8K.py": "Analyzes intervention results on GSM8K tasks",
    "run_activations.py": "Script for extracting model activations",
    "use_activations.py": "Script for using extracted activations",
    "run_batch_and_verify.py": "Batch processing utilities",
    "README.md": "Project documentation",
    "requirements.txt": "Project dependencies",
    "__init__.py": "Python package initialization file",
}

# Directories to ignore
IGNORE_DIRS = {".git", "__pycache__", "checkpoints"}

def get_file_description(file_path):
    """Get a description for a file based on its name or extension."""
    filename = os.path.basename(file_path)
    
    # Check if we have a specific description for this file
    if filename in FILE_DESCRIPTIONS:
        return FILE_DESCRIPTIONS[filename]
    
    # Otherwise, use extension-based description
    ext = os.path.splitext(filename)[1]
    if ext in EXTENSION_DESCRIPTIONS:
        return EXTENSION_DESCRIPTIONS[ext]
    
    return "Unknown file type"

def list_files(directory="."):
    """List all files in the directory and subdirectories with descriptions."""
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            # Skip certain files
            if file == ".DS_Store":
                continue
                
            file_path = os.path.join(root, file)
            description = get_file_description(file_path)
            
            results.append({
                "path": file_path,
                "description": description
            })
    
    return results

def main():
    """Main function to list all files."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    # List files in the src directory
    src_files = list_files(os.path.join(project_root, "src"))
    
    # Print results in a readable format
    print(f"{'File Path':<60} | {'Description':<40}")
    print(f"{'-' * 60} | {'-' * 40}")
    
    for file_info in sorted(src_files, key=lambda x: x["path"]):
        rel_path = os.path.relpath(file_info["path"], project_root)
        print(f"{rel_path:<60} | {file_info['description']:<40}")
    
    # Optionally save to a JSON file
    with open(os.path.join(project_root, "src", "file_listing.json"), "w") as f:
        json.dump(
            [{"path": os.path.relpath(f["path"], project_root), "description": f["description"]} for f in src_files],
            f, 
            indent=2
        )
    
    print(f"\nTotal files: {len(src_files)}")
    print(f"File listing saved to {os.path.join('src', 'file_listing.json')}")

if __name__ == "__main__":
    main() 