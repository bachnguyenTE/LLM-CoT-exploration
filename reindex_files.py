#!/usr/bin/env python3
import os
import re
import glob
import shutil
import argparse
from collections import defaultdict

def reindex_files(source_text_dir, source_raw_dir, target_text_dir=None, target_raw_dir=None, 
                 dry_run=False, preserve_originals=False, verbose=False, group_by_temp=True):
    """
    Reindex files from source directories to target directories with sequential indices.
    
    Args:
        source_text_dir: Directory containing text files
        source_raw_dir: Directory containing raw output files
        target_text_dir: Target directory for reindexed text files (if None, files are renamed in place)
        target_raw_dir: Target directory for reindexed raw files (if None, files are renamed in place)
        dry_run: If True, only print what would be done without making changes
        preserve_originals: If True, always copy files instead of renaming them
        verbose: If True, print more detailed information
        group_by_temp: If True, reindex each temperature group separately
    """
    # Ensure source directories exist
    if not os.path.exists(source_text_dir):
        raise FileNotFoundError(f"Text directory not found: {source_text_dir}")
    if not os.path.exists(source_raw_dir):
        raise FileNotFoundError(f"Raw outputs directory not found: {source_raw_dir}")
    
    # If preserve_originals is True, we need target directories
    if preserve_originals:
        if not target_text_dir:
            target_text_dir = f"{source_text_dir}_reindexed"
        if not target_raw_dir:
            target_raw_dir = f"{source_raw_dir}_reindexed"
    
    # Create target directories if they don't exist and are specified
    if target_text_dir and not os.path.exists(target_text_dir):
        if not dry_run:
            os.makedirs(target_text_dir, exist_ok=True)
        print(f"Created directory: {target_text_dir}")
    
    if target_raw_dir and not os.path.exists(target_raw_dir):
        if not dry_run:
            os.makedirs(target_raw_dir, exist_ok=True)
        print(f"Created directory: {target_raw_dir}")
    
    # Find all text files
    text_files = glob.glob(os.path.join(source_text_dir, "*.txt"))
    
    if verbose:
        print(f"Found {len(text_files)} text files in {source_text_dir}")
    
    # Extract indices and temperature values from text filenames
    file_info_by_temp = defaultdict(list)
    pattern = r'text_(\d+)_temp([\d_]+)\.txt'
    
    for text_file in text_files:
        filename = os.path.basename(text_file)
        match = re.match(pattern, filename)
        if match:
            index = int(match.group(1))
            temp = match.group(2)
            
            # Look for matching raw output file
            raw_filename = f"output_{index}_temp{temp}.pt"
            raw_file = os.path.join(source_raw_dir, raw_filename)
            
            if os.path.exists(raw_file):
                file_info = {
                    'index': index,
                    'temp': temp,
                    'text_file': text_file,
                    'raw_file': raw_file
                }
                
                if group_by_temp:
                    file_info_by_temp[temp].append(file_info)
                else:
                    # Use a default key if not grouping by temperature
                    file_info_by_temp['all'].append(file_info)
            else:
                print(f"Warning: No matching raw file found for {filename}")
        else:
            print(f"Warning: Filename format not recognized: {filename}")
    
    if not file_info_by_temp:
        print("No valid file pairs found for reindexing.")
        return
    
    total_pairs = sum(len(files) for files in file_info_by_temp.values())
    print(f"Found {total_pairs} valid file pairs across {len(file_info_by_temp)} temperature groups")
    
    # Sort each group by original index
    for temp, files in file_info_by_temp.items():
        files.sort(key=lambda x: x['index'])
    
    if verbose:
        for temp, files in file_info_by_temp.items():
            readable_temp = temp.replace('_', '.')
            print(f"\nTemperature {readable_temp}: {len(files)} files")
            print(f"  Original indices: {[info['index'] for info in files]}")
    
    # Create a summary of reindexing operations
    print(f"\nReindexing Plan:")
    print(f"{'Temperature':<15} {'Original Index':<15} {'New Index':<15}")
    print(f"{'-'*15:<15} {'-'*15:<15} {'-'*15:<15}")
    
    for temp, files in sorted(file_info_by_temp.items()):
        readable_temp = temp.replace('_', '.')
        for new_index, info in enumerate(files):
            orig_index = info['index']
            print(f"{readable_temp:<15} {orig_index:<15} {new_index:<15}")
    
    if dry_run:
        print("\nDry run - no changes made")
        return
    
    # Confirm with user if not in dry run mode
    if not dry_run:
        response = input("\nProceed with reindexing? [y/N]: ").strip().lower()
        if response != 'y':
            print("Reindexing cancelled.")
            return
    
    # Track renamed files to handle collisions
    renamed_files = set()
    
    # Process each temperature group
    for temp, files in file_info_by_temp.items():
        if verbose:
            readable_temp = temp.replace('_', '.')
            print(f"\nProcessing temperature group: {readable_temp}")
        
        # Reindex files within this temperature group
        for new_index, info in enumerate(files):
            orig_index = info['index']
            file_temp = info['temp']
            text_file = info['text_file']
            raw_file = info['raw_file']
            
            # Skip if these files have already been processed
            if text_file in renamed_files or raw_file in renamed_files:
                print(f"Skipping already processed files: {os.path.basename(text_file)}")
                continue
            
            # Generate new filenames
            new_text_filename = f"text_{new_index}_temp{file_temp}.txt"
            new_raw_filename = f"output_{new_index}_temp{file_temp}.pt"
            
            # Determine target paths
            if target_text_dir:
                new_text_path = os.path.join(target_text_dir, new_text_filename)
            else:
                new_text_path = os.path.join(os.path.dirname(text_file), new_text_filename)
                
            if target_raw_dir:
                new_raw_path = os.path.join(target_raw_dir, new_raw_filename)
            else:
                new_raw_path = os.path.join(os.path.dirname(raw_file), new_raw_filename)
            
            # Check for potential filename collisions
            if os.path.exists(new_text_path) and new_text_path != text_file:
                print(f"Warning: Target file already exists: {new_text_path}")
                continue
                
            if os.path.exists(new_raw_path) and new_raw_path != raw_file:
                print(f"Warning: Target file already exists: {new_raw_path}")
                continue
                
            # Copy or rename files
            try:
                # Always copy if preserve_originals is True or target dirs are specified
                should_copy = preserve_originals or target_text_dir or target_raw_dir
                
                if should_copy:
                    shutil.copy2(text_file, new_text_path)
                    shutil.copy2(raw_file, new_raw_path)
                    operation = "Copied"
                else:
                    # Rename in place if not preserving originals
                    os.rename(text_file, new_text_path)
                    os.rename(raw_file, new_raw_path)
                    operation = "Renamed"
                    renamed_files.add(text_file)
                    renamed_files.add(raw_file)
                
                # Print what was done
                if orig_index != new_index or verbose:
                    readable_temp = file_temp.replace('_', '.')
                    print(f"{operation}: {os.path.basename(text_file)} → {new_text_filename} (Temp: {readable_temp})")
                    print(f"{operation}: {os.path.basename(raw_file)} → {new_raw_filename}")
                    
            except Exception as e:
                print(f"Error processing files: {e}")
    
    print(f"\nReindexing complete: {total_pairs} pairs of files processed")
    
    if target_text_dir or target_raw_dir:
        if target_text_dir:
            print(f"Reindexed text files are in: {target_text_dir}")
        if target_raw_dir:
            print(f"Reindexed raw files are in: {target_raw_dir}")

def main():
    parser = argparse.ArgumentParser(description='Reindex penguin output files to sequential indices')
    parser.add_argument('--text-dir', default='outputs/penguin/decoded_text_reindexed',
                      help='Directory containing text files (default: outputs/penguin/decoded_text_reindexed)')
    parser.add_argument('--raw-dir', default='outputs/penguin/raw_outputs_reindexed',
                      help='Directory containing raw output files (default: outputs/penguin/raw_outputs_reindexed)')
    parser.add_argument('--target-text-dir', 
                      help='Target directory for reindexed text files (default: same as source)')
    parser.add_argument('--target-raw-dir', 
                      help='Target directory for reindexed raw files (default: same as source)')
    parser.add_argument('--preserve', action='store_true',
                      help='Preserve original files by copying instead of renaming')
    parser.add_argument('--no-group-by-temp', action='store_true',
                      help='Do not group files by temperature when reindexing')
    parser.add_argument('--dry-run', action='store_true',
                      help='Print what would be done without making changes')
    parser.add_argument('--verbose', action='store_true',
                      help='Print more detailed information')
    
    args = parser.parse_args()
    
    try:
        reindex_files(
            args.text_dir, 
            args.raw_dir, 
            args.target_text_dir, 
            args.target_raw_dir,
            args.dry_run,
            args.preserve,
            args.verbose,
            not args.no_group_by_temp
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 