import torch
import os
import glob
import json
import re
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

def extract_id(filepath):
    """Extract numeric ID from filename following pattern output_X_activations.pt"""
    match = re.search(r'output_(\d+)_activations\.pt', os.path.basename(filepath))
    return int(match.group(1)) if match else None

def check_file_integrity(file_path, verbose=True):
    """Check that a single activation file has all expected data with reasonable values"""
    if verbose:
        print(f"Checking file: {file_path}")
    
    try:
        # Load the activation file
        data = torch.load(file_path)
        
        # Check for expected keys
        expected_keys = ["attention_activations", "final_hidden_state", "logits", "input_ids", "model_name", "input_file"]
        missing_keys = [key for key in expected_keys if key not in data]
        
        if missing_keys:
            print(f"  ❌ Missing keys: {missing_keys}")
            return False
        
        # Check data shapes
        if verbose:
            print(f"  ✓ Input IDs shape: {data['input_ids'].shape}")
            print(f"  ✓ Final hidden state shape: {data['final_hidden_state'].shape}")
            print(f"  ✓ Logits shape: {data['logits'].shape}")
            print(f"  ✓ Number of attention layers: {len(data['attention_activations'])}")
        
        # Check for NaN values
        has_nans = False
        if torch.isnan(data['final_hidden_state']).any():
            print(f"  ❌ NaN values found in final_hidden_state")
            has_nans = True
        
        if torch.isnan(data['logits']).any():
            print(f"  ❌ NaN values found in logits")
            has_nans = True
        
        # Check attention activations
        for layer_idx, attn in data['attention_activations'].items():
            if torch.isnan(attn).any():
                print(f"  ❌ NaN values found in attention layer {layer_idx}")
                has_nans = True
                break
        
        # Check compatibility between inputs and outputs
        if data['final_hidden_state'].shape[0] != data['input_ids'].shape[0]:
            print(f"  ❌ Batch size mismatch: inputs={data['input_ids'].shape[0]}, outputs={data['final_hidden_state'].shape[0]}")
            return False
            
        if data['final_hidden_state'].shape[1] != data['input_ids'].shape[1]:
            print(f"  ❌ Sequence length mismatch: inputs={data['input_ids'].shape[1]}, outputs={data['final_hidden_state'].shape[1]}")
            return False
        
        # Check for metadata file
        metadata_path = file_path.replace(".pt", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if verbose:
                    print(f"  ✓ Metadata file exists")
                    print(f"  ✓ Model: {metadata['model_name']}")
        else:
            print(f"  ❌ Metadata file missing: {metadata_path}")
            return False
        
        # Check that the original input file exists
        if not os.path.exists(data['input_file']):
            print(f"  ⚠️ Original input file not found: {data['input_file']}")
        
        if not has_nans:
            if verbose:
                print(f"  ✓ No NaN values found")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return False

def check_directory(directory_path, pattern="output_*_activations.pt", sample_size=None, verbose=True):
    """Check all activation files in a directory"""
    print(f"Checking activations in {directory_path}")
    
    # Try to infer model and dataset from path
    try:
        path_parts = Path(directory_path).parts
        # The structure is expected to be activations/model_name/dataset_name
        if len(path_parts) >= 3:
            model_name = path_parts[-2]  # Second to last part is model
            dataset_name = path_parts[-1]  # Last part is dataset
            print(f"Detected model: {model_name}, dataset: {dataset_name}")
    except:
        pass  # If we can't infer, just continue
    
    # Find all activation files
    glob_pattern = os.path.join(directory_path, pattern)
    files = glob.glob(glob_pattern)
    files.sort(key=extract_id)
    
    if not files:
        print(f"No activation files found matching {glob_pattern}")
        return
    
    print(f"Found {len(files)} activation files")
    
    # Sample if requested
    if sample_size and sample_size < len(files):
        import random
        files = random.sample(files, sample_size)
        print(f"Checking a sample of {sample_size} files")
    
    # Check each file
    success_count = 0
    failed_files = []
    
    for file_path in tqdm(files, desc="Checking files"):
        file_success = check_file_integrity(file_path, verbose=verbose)
        if file_success:
            success_count += 1
        else:
            failed_files.append(file_path)
    
    # Report results
    print(f"\nSummary:")
    print(f"  ✓ {success_count}/{len(files)} files passed integrity checks")
    
    if failed_files:
        print(f"  ❌ {len(failed_files)} files failed checks:")
        for file in failed_files[:5]:  # Show first 5 failures
            print(f"    - {os.path.basename(file)}")
        if len(failed_files) > 5:
            print(f"    ... and {len(failed_files) - 5} more")

def main():
    parser = argparse.ArgumentParser(description='Check the integrity of model activation files')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing activation files')
    parser.add_argument('--pattern', type=str, default="output_*_activations.pt",
                        help='Glob pattern to match activation files')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of files to sample for checking (default: check all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information for each file')
    
    args = parser.parse_args()
    
    check_directory(
        directory_path=args.dir,
        pattern=args.pattern,
        sample_size=args.sample,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 