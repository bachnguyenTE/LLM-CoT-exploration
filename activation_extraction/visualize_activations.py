import torch
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def load_activation_file(file_path):
    """Load activation data from a file and return basic information"""
    print(f"Loading activation file: {file_path}")
    
    try:
        # Load the data
        data = torch.load(file_path)
        
        # Extract key information
        model_name = data['model_name']
        input_file = data['input_file']
        input_ids = data['input_ids']
        final_hidden_state = data['final_hidden_state']
        logits = data['logits']
        attention_activations = data['attention_activations']
        
        # Print basic information
        print(f"Model: {model_name}")
        print(f"Original input file: {input_file}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Final hidden state shape: {final_hidden_state.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Number of attention layers: {len(attention_activations)}")
        
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def visualize_attention_patterns(data, layer_idx=0, head_idx=0, max_seq_len=50, save_path=None):
    """Visualize attention patterns for a specific layer and head"""
    if 'attention_activations' not in data:
        print("No attention activations found in data")
        return
    
    attention_activations = data['attention_activations']
    
    if layer_idx not in attention_activations:
        print(f"Layer {layer_idx} not found in attention activations")
        return
    
    # Get attention activations for the specified layer
    layer_activations = attention_activations[layer_idx]
    
    # Extract the specified head
    num_heads = layer_activations.shape[2]
    if head_idx >= num_heads:
        print(f"Head index {head_idx} out of range (max: {num_heads-1})")
        return
    
    # Extract attention for the first batch item, specified head
    attention = layer_activations[0, :, head_idx, :]
    
    # Convert from BFloat16 to float32 if needed
    if attention.dtype == torch.bfloat16:
        print(f"Converting attention tensor from {attention.dtype} to float32")
        attention = attention.to(torch.float32)
    
    # Limit sequence length for visualization
    seq_len = min(attention.shape[0], max_seq_len)
    attention = attention[:seq_len, :seq_len]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention.numpy(), cmap='viridis', annot=False, xticklabels=5, yticklabels=5)
    plt.title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Token Position (Target)")
    plt.ylabel("Token Position (Source)")
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def visualize_hidden_state_pca(data, max_tokens=100, save_path=None):
    """Visualize PCA of hidden states to see token embeddings"""
    from sklearn.decomposition import PCA
    
    if 'final_hidden_state' not in data:
        print("No final hidden state found in data")
        return
    
    # Get hidden states for the first batch item
    hidden_states = data['final_hidden_state'][0]
    
    # Convert from BFloat16 to float32 if needed
    if hidden_states.dtype == torch.bfloat16:
        print(f"Converting hidden states tensor from {hidden_states.dtype} to float32")
        hidden_states = hidden_states.to(torch.float32)
    
    # Limit number of tokens for visualization
    num_tokens = min(hidden_states.shape[0], max_tokens)
    hidden_states = hidden_states[:num_tokens]
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(hidden_states.numpy())
    
    # Plot the reduced representations
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.8)
    
    # Add sequential numbers as labels
    for i in range(num_tokens):
        plt.annotate(str(i), (reduced[i, 0], reduced[i, 1]))
    
    plt.title("PCA of Token Hidden States")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"PCA visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_top_logits(data, tokenizer, top_k=5):
    """Visualize the top predicted tokens at each position"""
    if 'logits' not in data or 'input_ids' not in data:
        print("Logits or input IDs not found in data")
        return
    
    logits = data['logits'][0]  # First batch item
    input_ids = data['input_ids'][0]  # First batch item
    
    # Convert from BFloat16 to float32 if needed
    if logits.dtype == torch.bfloat16:
        print(f"Converting logits tensor from {logits.dtype} to float32")
        logits = logits.to(torch.float32)
    
    # Get top k predictions for each position
    top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
    
    # Convert to probabilities
    top_probs = torch.softmax(top_values, dim=-1)
    
    # Print top predictions for a few positions
    max_pos = min(10, logits.shape[0])  # Show at most 10 positions
    
    print("\nTop token predictions:")
    for pos in range(max_pos):
        input_token = tokenizer.decode(input_ids[pos:pos+1])
        print(f"\nPosition {pos} (Input: '{input_token}'):")
        
        for i in range(top_k):
            token_id = top_indices[pos, i].item()
            token = tokenizer.decode([token_id])
            prob = top_probs[pos, i].item() * 100
            print(f"  {i+1}. '{token}' ({prob:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Visualize model activation data')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to activation file (.pt)')
    parser.add_argument('--layer', type=int, default=0,
                        help='Layer index for attention visualization')
    parser.add_argument('--head', type=int, default=0,
                        help='Head index for attention visualization')
    parser.add_argument('--mode', type=str, choices=['attention', 'pca', 'logits', 'all'], default='all',
                        help='Visualization mode')
    parser.add_argument('--output_dir', type=str, default='viz_output',
                        help='Directory to save visualizations')
    parser.add_argument('--save', action='store_true',
                        help='Save visualizations to files instead of displaying')
    
    args = parser.parse_args()
    
    # Load the activation file
    data = load_activation_file(args.file)
    if data is None:
        return
    
    # Get the model name and load the tokenizer for token decoding
    model_name = data['model_name']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine appropriate directories
    base_filename = os.path.basename(args.file).replace('.pt', '')
    
    # Use the provided output_dir directly - we expect it to already have the model/dataset structure
    # if it was passed from run_batch_and_verify.py
    viz_dir = args.output_dir
    
    # Run visualizations based on the mode
    if args.mode in ['attention', 'all']:
        if args.save:
            save_path = os.path.join(viz_dir, f"{base_filename}_attn_layer{args.layer}_head{args.head}.png")
        else:
            save_path = None
        visualize_attention_patterns(data, args.layer, args.head, save_path=save_path)
    
    if args.mode in ['pca', 'all']:
        if args.save:
            save_path = os.path.join(viz_dir, f"{base_filename}_pca.png")
        else:
            save_path = None
        visualize_hidden_state_pca(data, save_path=save_path)
    
    if args.mode in ['logits', 'all']:
        visualize_top_logits(data, tokenizer)

if __name__ == "__main__":
    main() 