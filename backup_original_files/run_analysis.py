#!/usr/bin/env python3
import os
import sys
import argparse
from parse_penguin_results import main as parse_results
from visualize_results import visualize_results

def main():
    parser = argparse.ArgumentParser(description='Analyze penguin classification results')
    parser.add_argument('--parse-only', action='store_true', help='Only parse results without visualization')
    parser.add_argument('--visualize-only', action='store_true', help='Only create visualizations without parsing')
    
    args = parser.parse_args()
    
    # Check if output directory exists
    output_dir = 'outputs/penguin/decoded_text_reindexed'
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        print("Please run penguin_generator.py first to generate model outputs.")
        return 1
    
    # Check if there are any text files
    if len(os.listdir(output_dir)) == 0:
        print(f"Error: No text files found in '{output_dir}'.")
        print("Please run penguin_generator.py first to generate model outputs.")
        return 1
    
    # Run analysis based on arguments
    if args.visualize_only:
        print("Creating visualizations...")
        visualize_results()
    elif args.parse_only:
        print("Parsing results...")
        parse_results()
    else:
        print("Parsing results...")
        parse_results()
        print("\nCreating visualizations...")
        visualize_results()
    
    print("\nAnalysis complete!")
    if not args.parse_only:
        print("Visualizations have been saved to 'outputs/visualizations/'.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 