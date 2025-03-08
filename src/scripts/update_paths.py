#!/usr/bin/env python3
"""
Script to update paths in shell scripts to match the new directory structure.
This script adds a SCRIPT_DIR variable to shell scripts and updates paths to use it.
"""

import os
import re
import glob
import shutil
from pathlib import Path

def update_shell_script(filepath):
    """Update a shell script to use SCRIPT_DIR for Python script paths."""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Add SCRIPT_DIR if it doesn't exist
    if "SCRIPT_DIR=" not in content:
        script_dir_line = '\n# Get the directory where this script is located\nSCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n'
        
        # Insert after the argument parsing section
        if "fi" in content:
            # Find the last "fi" before the main script logic
            parts = content.split("fi", 1)
            if len(parts) > 1:
                content = parts[0] + "fi" + script_dir_line + parts[1]
            else:
                # If no "fi" found, insert after the shebang
                content = re.sub(r'(#!/bin/bash.*?\n)', r'\1' + script_dir_line, content, flags=re.DOTALL)
        else:
            # If no "fi" found, insert after the shebang
            content = re.sub(r'(#!/bin/bash.*?\n)', r'\1' + script_dir_line, content, flags=re.DOTALL)
    
    # Update Python script paths
    content = re.sub(
        r'python\s+([a-zA-Z0-9_]+\.py)',
        r'python "$SCRIPT_DIR/\1"',
        content
    )
    
    with open(filepath, 'w') as file:
        file.write(content)
    
    print(f"Updated {filepath}")

def main():
    """Update all shell scripts in the src directory."""
    src_dir = Path(__file__).parent.parent
    shell_scripts = list(src_dir.glob('**/*.sh'))
    
    for script in shell_scripts:
        update_shell_script(script)
    
    print(f"Updated {len(shell_scripts)} shell scripts")

if __name__ == "__main__":
    main() 