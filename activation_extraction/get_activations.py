import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import json
import glob
import re
from pathlib import Path
from tqdm import tqdm
import time
import warnings

# Suppress ALL warnings globally to keep output clean
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

# Try to suppress specific PyTorch and HuggingFace warnings
try:
    from transformers import logging
    logging.set_verbosity_error()  # Only show errors, not warnings
except:
    pass

# Set PyTorch warnings to a minimum
torch.set_warn_always(False)

def extract_attention_heads(model, inputs):
    """
    Extracts the attention head outputs for each transformer layer.
    Assumes each layer has an attribute `self_attn` (the self-attention module).
    The self-attention module produces an output of shape [B, T, hidden_size],
    which is then reshaped into [B, T, num_heads, head_dim].
    
    Args:
        model: The transformer model.
        inputs: A dictionary containing model inputs (e.g., {"input_ids": tensor}).
        
    Returns:
        A dictionary mapping layer indices to their attention head activations.
    """
    attention_activations = {}
    hook_handles = []

    def get_attn_hook(layer_idx):
        def hook(module, layer_input, layer_output):
            # If layer_output is a tuple, extract the first element
            layer_output_tensor = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # Expected shape: [B, T, hidden_size]
            batch_size, seq_length, hidden_size = layer_output_tensor.shape
            # Use the module attribute 'num_heads' if available
            num_heads = module.num_heads if hasattr(module, "num_heads") else 1
            head_dim = hidden_size // num_heads
            # Reshape to [B, T, num_heads, head_dim]
            attn_heads = layer_output_tensor.view(batch_size, seq_length, num_heads, head_dim)
            attention_activations[layer_idx] = attn_heads.detach().cpu()
        return hook

    # Set up progress tracking
    num_layers = len(model.model.layers)
    
    # Register a hook on each transformer layer's self-attention module (assumed at layer.self_attn)
    for idx, layer in enumerate(model.model.layers):
        handle = layer.self_attn.register_forward_hook(get_attn_hook(idx))
        hook_handles.append(handle)

    # Run the forward pass (no gradients needed)
    with torch.no_grad():
        _ = model(**inputs)

    # Remove hooks to avoid interference with future runs
    for handle in hook_handles:
        handle.remove()

    return attention_activations

def process_model_activations(model_name, input_file, output_dir, output_name=None, verbose=False, silent=False):
    """
    Process a model and extract activations for the given input.
    
    Args:
        model_name: Name of the HuggingFace model to use
        input_file: Path to the raw outputs file
        output_dir: Directory to save the activations to
        output_name: Name of the output file (default: model_activations.pt)
        verbose: Whether to print detailed information
        silent: Whether to suppress all output
    """
    # Set device (using "mps" for Apple Silicon, "cuda" for NVIDIA, or CPU as fallback)
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose and not silent:
        print(f"Loading data from: {input_file}")
    
    # Suppress warnings during processing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Step 1: Load data
        raw_outputs = torch.load(input_file, map_location="cpu")
        
        # Extract token IDs: if raw_outputs is a dict containing "input_ids", use that; otherwise assume it's a tensor
        if isinstance(raw_outputs, dict) and "input_ids" in raw_outputs:
            input_ids = raw_outputs["input_ids"]
        else:
            input_ids = raw_outputs
        
        if verbose and not silent:
            print("Input IDs shape:", input_ids.shape)
            # Decode the first sequence of token IDs into a human-readable sentence
            decoded_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"Decoded sentence (first 100 chars): {decoded_sentence[:100]}...")
        
        # Prepare inputs for the model (move them to the appropriate device)
        inputs = {"input_ids": input_ids.to(device)}
        
        # Step 2: Extract attention head activations from each transformer layer
        if verbose and not silent:
            print("Extracting attention head activations...")
        
        attention_activations = extract_attention_heads(model, inputs)
        
        if verbose and not silent:
            for layer_idx, attn_tensor in list(attention_activations.items())[:2]:  # Show only first two for brevity
                print(f"Layer {layer_idx} attention head activation shape: {attn_tensor.shape}")
            if len(attention_activations) > 2:
                print(f"... and {len(attention_activations) - 2} more layers")
        
        # Step 3: Extract final hidden state and logits
        if verbose and not silent:
            print("Extracting final hidden states and logits...")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states is a tuple containing the embeddings plus the output of each layer.
            # The final hidden state is the output of the last transformer block.
            final_hidden_state = outputs.hidden_states[-1]
            
            # Extract logits: either via dictionary key "logits" or the first element if tuple.
            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs[0]
        
        # Move final hidden state and logits to CPU and print their shapes.
        final_hidden_state = final_hidden_state.detach().cpu()
        logits = logits.detach().cpu()
        
        if verbose and not silent:
            print("Final hidden state shape:", final_hidden_state.shape)
            print("Final pre-softmax logits shape:", logits.shape)
        
        # Create a dictionary to store all activations and logits
        all_activations = {
            "attention_activations": attention_activations,
            "final_hidden_state": final_hidden_state,
            "logits": logits,
            "input_ids": input_ids,
            "model_name": model_name,
            "input_file": input_file
        }
        
        # Determine output file name
        if output_name is None:
            # Extract dataset and sample information from the input path
            input_path = Path(input_file)
            # Use parent directory name (usually the dataset name) and file name without extension
            file_name = input_path.stem
            output_name = f"{file_name}_activations.pt"
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Full output path
        output_path = os.path.join(output_dir, output_name)
        
        # Step 4: Save the dictionary as a .pt file
        if verbose and not silent:
            print(f"Saving activations to: {output_path}")
        
        torch.save(all_activations, output_path)
        
        # Save a metadata file with information about the extraction
        metadata = {
            "model_name": model_name,
            "input_file": input_file,
            "device": str(device),
            "shapes": {
                "input_ids": list(input_ids.shape),
                "final_hidden_state": list(final_hidden_state.shape),
                "logits": list(logits.shape),
                "num_layers": len(attention_activations)
            }
        }
        
        metadata_path = output_path.replace(".pt", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose and not silent:
            print(f"Metadata saved to {metadata_path}")
    
    return output_path

def process_directory(model_name, input_dir, output_dir, num_files=None, file_pattern="output_*.pt", verbose=False):
    """
    Process multiple files from a directory.
    
    Args:
        model_name: Name of the HuggingFace model to use
        input_dir: Directory containing raw output files
        output_dir: Directory to save activations to
        num_files: Number of files to process (None for all files)
        file_pattern: Glob pattern to match files
        verbose: Whether to print detailed information
    """
    # Set up model and tokenizer once for all files
    print(f"Loading model: {model_name}")
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    global model, tokenizer
    
    # Suppress PyTorch warnings during loading to keep the output clean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Load model and tokenizer without progress bars to avoid cluttering the output
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get list of files matching the pattern
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    
    # Sort files by numeric ID in filename
    def extract_id(filepath):
        match = re.search(r'output_(\d+)\.pt', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    file_paths.sort(key=extract_id)
    
    # Limit to num_files if specified
    if num_files is not None:
        file_paths = file_paths[:num_files]
    
    total_files = len(file_paths)
    print(f"Found {total_files} files to process")
    
    # Process each file with a tqdm progress bar
    processed_files = []
    
    # Start timing for overall processing time calculation
    start_time = time.time()
    
    # Use tqdm to create a single progress bar that updates for each file
    for file_path in tqdm(file_paths, desc="Processing files", total=total_files):
        file_id = extract_id(file_path)
        file_name = os.path.basename(file_path)
        
        output_name = f"output_{file_id}_activations.pt"
        output_path = process_model_activations(
            model_name=model_name,
            input_file=file_path,
            output_dir=output_dir,
            output_name=output_name,
            verbose=verbose
        )
        processed_files.append(output_path)
    
    # Calculate the total elapsed time
    total_time = time.time() - start_time
    avg_time_per_file = total_time / total_files if total_files > 0 else 0
    
    print(f"\nProcessed {len(processed_files)} files in {total_time:.2f}s ({avg_time_per_file:.2f}s/file)")
    return processed_files

def main():
    parser = argparse.ArgumentParser(description='Extract model activations from a transformer model')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help='Name of the HuggingFace model to use')
    parser.add_argument('--input', type=str, 
                        help='Path to the raw outputs file containing token IDs (for single file mode)')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing raw output files (for batch mode)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (used for organizing outputs, optional)')
    parser.add_argument('--output_dir', type=str, default="activations",
                        help='Directory to save the activations to')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Name of the output file (default: auto-generated based on input)')
    parser.add_argument('--num_files', type=int, default=None,
                        help='Number of files to process (default: process all files)')
    parser.add_argument('--file_pattern', type=str, default="output_*.pt",
                        help='Glob pattern to match files in input_dir')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during processing')
    
    args = parser.parse_args()
    
    # Extract a simplified model name for the directory
    model_short_name = args.model.split('/')[-1]
    
    # Create a nested directory structure: output_dir/model/dataset/
    if args.dataset:
        output_dir = os.path.join(args.output_dir, model_short_name, args.dataset)
    else:
        output_dir = os.path.join(args.output_dir, model_short_name)
    
    # Determine if we're processing a single file or multiple files
    if args.input_dir:
        # Batch mode
        process_directory(
            model_name=args.model,
            input_dir=args.input_dir,
            output_dir=output_dir,
            num_files=args.num_files,
            file_pattern=args.file_pattern,
            verbose=args.verbose
        )
    elif args.input:
        # Single file mode
        # Load the model and tokenizer
        global model, tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Process the model and extract activations
        process_model_activations(
            model_name=args.model,
            input_file=args.input,
            output_dir=output_dir,
            output_name=args.output_name,
            verbose=args.verbose
        )
    else:
        parser.error("Either --input or --input_dir must be specified")

if __name__ == "__main__":
    main()