import os
import argparse
import subprocess
import time
import glob
from pathlib import Path

def run_command(cmd, description, capture_output=False):
    """Run a command and handle any errors"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    if capture_output:
        # Capture output for commands where we want to process the output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print stdout with proper indentation
        if result.stdout:
            for line in result.stdout.splitlines():
                print(f"  {line}")
    else:
        # Don't capture output for commands that show progress bars
        result = subprocess.run(cmd)
    
    elapsed_time = time.time() - start_time
    print(f"Command completed in {elapsed_time:.2f} seconds")
    
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        if capture_output and result.stderr:
            print(f"Error output: {result.stderr}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run batch activation extraction and verify results')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help='Name of the HuggingFace model to use')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw output files')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (used for organizing outputs)')
    parser.add_argument('--output_dir', type=str, default="activations",
                        help='Directory to save the activations to')
    parser.add_argument('--num_files', type=int, default=3,
                        help='Number of files to process')
    parser.add_argument('--file_pattern', type=str, default="output_*.pt",
                        help='Glob pattern to match files in input_dir')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip extraction phase and only run verification')
    parser.add_argument('--skip_verification', action='store_true',
                        help='Skip verification phase and only run extraction')
    parser.add_argument('--viz_dir', type=str, default="viz_output",
                        help='Directory to save visualizations')
    
    # Add visualization control parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Run visualization on processed files')
    parser.add_argument('--viz_files', type=int, default=1,
                        help='Number of files to visualize (default: 1)')
    parser.add_argument('--viz_layers', type=str, default="0,10,20",
                        help='Comma-separated list of layers to visualize attention for')
    parser.add_argument('--viz_heads', type=str, default="0,4",
                        help='Comma-separated list of attention heads to visualize')
    parser.add_argument('--viz_modes', type=str, default="attention",
                        choices=["attention", "pca", "logits", "all"],
                        help='Visualization modes to run (default: attention)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during processing')
    
    args = parser.parse_args()
    
    # Extract a simplified model name for the directory
    model_short_name = args.model.split('/')[-1]
    
    # Create the full output directory path with model-centric structure
    full_output_dir = os.path.join(args.output_dir, model_short_name, args.dataset)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Create visualization directory with model-centric structure
    viz_output_dir = os.path.join(args.viz_dir, model_short_name, args.dataset)
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Phase 1: Extract activations
    if not args.skip_extraction:
        cmd = [
            "python", os.path.join(script_dir, "get_activations.py"),
            "--model", args.model,
            "--input_dir", args.input_dir,
            "--dataset", args.dataset,
            "--output_dir", args.output_dir,
            "--num_files", str(args.num_files),
            "--file_pattern", args.file_pattern
        ]
        
        if args.verbose:
            cmd.append("--verbose")
        
        # Don't capture output so progress bars show properly
        success = run_command(cmd, "Extracting Activations", capture_output=False)
        if not success:
            print("Activation extraction failed. Exiting.")
            return
    
    # Phase 2: Verify results
    if not args.skip_verification:
        # First run the integrity check
        verify_cmd = [
            "python", os.path.join(script_dir, "check_activations.py"),
            "--dir", full_output_dir,
            "--pattern", "output_*_activations.pt",
            "--sample", str(min(args.num_files, 3)),  # Check up to 3 files
            "--verbose"
        ]
        
        # Capture output for verification to show the formatted results
        success = run_command(verify_cmd, "Verifying Activation Files", capture_output=True)
        if not success:
            print("Activation verification failed.")
    
    # Phase 3: Visualize results (only if requested)
    if args.visualize:
        # Get activation files
        activation_files = glob.glob(os.path.join(full_output_dir, "output_*_activations.pt"))
        activation_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))  # Sort by file ID
        
        # Limit to the requested number of files
        if args.viz_files and args.viz_files < len(activation_files):
            activation_files = activation_files[:args.viz_files]
        
        if not activation_files:
            print("No activation files found for visualization.")
            return
        
        print(f"\n=== Visualizing {len(activation_files)} files ===")
        
        # Parse layers and heads to visualize
        viz_layers = [int(l) for l in args.viz_layers.split(',')]
        viz_heads = [int(h) for h in args.viz_heads.split(',')]
        
        # Run visualizations for each file and specified layers/heads
        for file_idx, file_path in enumerate(activation_files):
            file_id = Path(file_path).stem.split('_')[1]
            
            if args.viz_modes in ["attention", "all"]:
                for layer_idx in viz_layers:
                    for head_idx in viz_heads:
                        visualize_cmd = [
                            "python", os.path.join(script_dir, "visualize_activations.py"),
                            "--file", file_path,
                            "--mode", "attention",
                            "--layer", str(layer_idx),
                            "--head", str(head_idx),
                            "--save",
                            "--output_dir", viz_output_dir
                        ]
                        
                        # Capture output for visualization to show the nicely formatted information
                        run_command(
                            visualize_cmd, 
                            f"Visualizing Attention: File {file_id}, Layer {layer_idx}, Head {head_idx}",
                            capture_output=True
                        )
            
            if args.viz_modes in ["pca", "all"]:
                pca_cmd = [
                    "python", os.path.join(script_dir, "visualize_activations.py"),
                    "--file", file_path,
                    "--mode", "pca",
                    "--save",
                    "--output_dir", viz_output_dir
                ]
                
                run_command(pca_cmd, f"Visualizing PCA: File {file_id}", capture_output=True)
            
            if args.viz_modes in ["logits", "all"]:
                logits_cmd = [
                    "python", os.path.join(script_dir, "visualize_activations.py"),
                    "--file", file_path,
                    "--mode", "logits",
                    "--output_dir", viz_output_dir
                ]
                
                run_command(logits_cmd, f"Showing Token Predictions: File {file_id}", capture_output=True)
    
    print("\n=== All operations completed ===")
    print(f"Activations saved to: {full_output_dir}")
    if args.visualize:
        print(f"Visualizations saved to: {viz_output_dir}")

if __name__ == "__main__":
    main() 