import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import pickle
from scipy.stats import gaussian_kde
import gc

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: float = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.encoder(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)

    @classmethod
    def from_pretrained(cls, path: str, input_dim: int, expansion_factor: float = 16, device: str = "mps") -> "SparseAutoencoder":
        model = cls(input_dim=input_dim, expansion_factor=expansion_factor)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model


def gather_residual_activations(model, target_layer, inputs):
    """
    Get activations from the model's residual stream at a specific layer
    """
    activations = []
    
    def hook_fn(mod, inputs, outputs):
        # Get the input to the layer, which is the residual stream
        # Make a copy of the tensor to avoid potential memory issues
        activations.append(inputs[0].detach().clone())
    
    # Register hook on the input_layernorm of the target layer
    # This is before any processing, so we get the raw residual stream
    hook = model.model.layers[target_layer].input_layernorm.register_forward_hook(hook_fn)
    
    # Run the model
    with torch.no_grad():
        model(inputs)
    
    # Remove the hook to avoid memory leaks
    hook.remove()
    
    # Return the stored activations
    if activations:
        return activations[0]  # Return the first batch element
    else:
        raise ValueError("No activations were captured by the hook")


def ensure_same_device(sae, target_act):
    """Ensure SAE and activations are on the same device"""
    model_device = target_act.device
    sae = sae.to(model_device)
    return sae, target_act.to(model_device)


def analyze_specific_tokens(model, tokenizer, inputs, target_tokens):
    """Analyze specific tokens' positions and relationships in the text"""
    results = {
        'token_positions': defaultdict(list),  # Where tokens appear in sequence
        'token_pairs': defaultdict(int)        # Which tokens appear together
    }
    
    # Get the full text and decoded tokens
    full_text = tokenizer.decode(inputs[0])
    decoded_tokens = tokenizer.batch_decode([[token.item()] for token in inputs[0]])
    
    # Find thinking section boundaries
    thinking_start = full_text.find("<think>")
    thinking_end = full_text.find("</think>")
    
    if thinking_start != -1 and thinking_end != -1:
        # Convert character positions to token indices
        thinking_start_idx = None
        thinking_end_idx = None
        cumulative_text = ""
        
        for i, token in enumerate(decoded_tokens):
            cumulative_text += token
            
            if thinking_start_idx is None and len(cumulative_text) > thinking_start:
                thinking_start_idx = i
            
            if thinking_start_idx is not None and thinking_end_idx is None and len(cumulative_text) > thinking_end:
                thinking_end_idx = i
                break
        
        # Only analyze tokens within the thinking section
        if thinking_start_idx is not None and thinking_end_idx is not None:
            # Analyze token positions and relationships within thinking section
            window_size = 5  # Look for token pairs within this window
            for i in range(thinking_start_idx, thinking_end_idx):
                token = decoded_tokens[i].strip()
                if token in target_tokens:
                    # Record position relative to thinking section start
                    results['token_positions'][token].append(i - thinking_start_idx)
                    
                    # Look for pairs within window
                    start = max(thinking_start_idx, i - window_size)
                    end = min(thinking_end_idx, i + window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            other_token = decoded_tokens[j].strip()
                            if other_token in target_tokens:
                                pair = tuple(sorted([token, other_token]))
                                results['token_pairs'][pair] += 1
    
    return results


def aggregate_token_analysis(all_results, target_tokens, token_to_top_features=None):
    """Aggregate token analysis results across all files"""
    aggregated = {
        'token_frequencies': Counter(),  # Total occurrences across all files
        'token_file_presence': Counter(),  # Number of files where token appears
        'token_pairs': Counter(),
        'token_stats': defaultdict(lambda: {
            'total_occurrences': 0,
            'files_present': 0,
            'most_common_pairs': Counter(),
            'top_features': defaultdict(list)  # Store feature activations for each token
        })
    }
    
    total_files = len(all_results)
    
    for result in all_results:
        # Track which tokens appear in this file
        tokens_in_file = set()
        
        # Update token frequencies
        for token in target_tokens:
            positions = result['token_positions'].get(token, [])
            if positions:  # Token appears in this file
                tokens_in_file.add(token)
                # Update total occurrences
                occurrences = len(positions)
                aggregated['token_frequencies'][token] += occurrences
                aggregated['token_stats'][token]['total_occurrences'] += occurrences
        
        # Update file presence counts
        for token in tokens_in_file:
            aggregated['token_file_presence'][token] += 1
            aggregated['token_stats'][token]['files_present'] += 1
        
        # Update token pairs
        for pair, count in result['token_pairs'].items():
            aggregated['token_pairs'][pair] += count
            # Update most common pairs for each token
            for token in pair:
                other_token = pair[1] if pair[0] == token else pair[0]
                aggregated['token_stats'][token]['most_common_pairs'][other_token] += count
    
    # Add feature activation information if provided
    if token_to_top_features:
        for token in target_tokens:
            if token in token_to_top_features:
                # Count feature occurrences and sum activations
                feature_counts = Counter()
                feature_total_activations = defaultdict(float)
                
                for feat_id, activation in token_to_top_features[token]:
                    feature_counts[feat_id] += 1
                    feature_total_activations[feat_id] += activation
                
                # Calculate average activation for each feature
                for feat_id in feature_counts:
                    avg_activation = feature_total_activations[feat_id] / feature_counts[feat_id]
                    aggregated['token_stats'][token]['top_features'][feat_id] = {
                        'count': feature_counts[feat_id],
                        'avg_activation': avg_activation
                    }
    
    return aggregated


def analyze_all_penguin_files(model, tokenizer, sae, layer_id=19, device="mps", max_files=None, target_tokens=None, activation_threshold=0.15, input_folder="outputs/penguin/raw_outputs", output_folder="outputs/analysis"):
    # Get all penguin output files
    penguin_files = glob.glob(os.path.join(input_folder, "*.pt"))
    
    if max_files is not None:
        penguin_files = penguin_files[:max_files]
    
    print(f"Analyzing {len(penguin_files)} penguin files...")

    # Dictionary to store aggregated feature activations
    file_features = {}
    token_to_top_features = defaultdict(list)
    feature_to_tokens = defaultdict(list)
    file_answers = {}  # Store YES/NO answers for each file
    
    # List to store token-specific analysis results
    token_analysis_results = []
    
    # Count successful files
    successful_files = 0

    # Process each file and analyze activations
    for pt_file_path in tqdm(penguin_files):
        try:
            # Extract file ID
            file_id = os.path.basename(pt_file_path).replace("output_", "").replace(".pt", "")
            
            # Load the tokens
            loaded_tokens = torch.load(pt_file_path, weights_only=True)
            
            # Ensure tokens are on the right device
            inputs = loaded_tokens.to(device)
            
            # Get the full text and check for YES/NO answer
            full_text = tokenizer.decode(inputs[0])
            
            # Find the last instance of YES and NO
            last_yes_pos = full_text.rfind("YES")
            last_no_pos = full_text.rfind("NO")
            
            if last_yes_pos != -1 and (last_no_pos == -1 or last_yes_pos > last_no_pos):
                file_answers[file_id] = 'YES'
            elif last_no_pos != -1:
                file_answers[file_id] = 'NO'
            
            # If target tokens specified, analyze their patterns
            if target_tokens:
                token_results = analyze_specific_tokens(model, tokenizer, inputs, target_tokens)
                token_results['file_id'] = file_id  # Add file ID to results
                token_analysis_results.append(token_results)
            
            # Get activations from the model's layer
            try:
                # Get the activations at the layer
                activations = gather_residual_activations(model, layer_id, inputs)
                
                # Ensure SAE and activations are on the same device
                sae, activations = ensure_same_device(sae, activations)
                
                # Get the decoded tokens for reference
                decoded_tokens = tokenizer.batch_decode([[token.item()] for token in inputs[0]])
                
                # Encode activations with SAE to get sparse features
                with torch.no_grad():
                    # We need to handle shape correctly - activations are [seq_len, batch, hidden_dim]
                    # but SAE expects [batch, seq_len, hidden_dim]
                    if len(activations.shape) == 3:  # [batch, seq_len, hidden_dim]
                        features = sae.encode(activations.float())
                    else:  # [seq_len, hidden_dim] or other shape
                        # Reshape to expected format
                        reshaped_activations = activations.float().unsqueeze(0) if len(activations.shape) == 2 else activations.float()
                        features = sae.encode(reshaped_activations)
                
                # Calculate mean activation per feature across tokens
                mean_activations = features.mean(dim=1).squeeze().cpu().numpy()
                
                # Store the top activated features for this file
                top_features_indices = np.argsort(mean_activations)[-20:][::-1]
                top_features_values = {int(idx): float(mean_activations[idx]) for idx in top_features_indices if mean_activations[idx] > activation_threshold}
                file_features[file_id] = top_features_values
                
                # Map tokens to their top features and vice versa
                for i, token_id in enumerate(inputs[0]):
                    if i >= features.size(1):  # Skip if index is out of bounds
                        continue
                        
                    token = decoded_tokens[i].strip()
                    if not token:  # Skip empty tokens
                        continue
                    
                    # Get top features for this token
                    token_features = features[0, i].cpu().numpy()
                    token_top_features = np.argsort(token_features)[-5:][::-1]  # Get top 5 features
                    
                    # Only keep features above threshold
                    significant_features = [(int(idx), float(token_features[idx])) for idx in token_top_features 
                                             if token_features[idx] > activation_threshold]
                    
                    # Store token -> features mapping
                    if significant_features:
                        token_to_top_features[token].extend(significant_features)
                        
                        # Store feature -> tokens mapping
                        for feat_idx, feat_val in significant_features:
                            feature_to_tokens[feat_idx].append((token, feat_val))
                
                # Increment successful file count
                successful_files += 1
                
                # Clear memory after processing each file
                del inputs, loaded_tokens, activations, features, mean_activations
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing activations for {pt_file_path}: {str(e)}")
        
        except Exception as e:
            print(f"Error loading file {pt_file_path}: {str(e)}")

    print(f"Successfully processed {successful_files} out of {len(penguin_files)} files")
    
    # Count YES/NO distribution
    yes_count = sum(1 for answer in file_answers.values() if answer == 'YES')
    no_count = sum(1 for answer in file_answers.values() if answer == 'NO')
    total_answers = len(file_answers)
    
    print(f"\nAnswer distribution:")
    if total_answers > 0:
        print(f"YES: {yes_count} files ({(yes_count/total_answers)*100:.1f}%)")
        print(f"NO: {no_count} files ({(no_count/total_answers)*100:.1f}%)")
    else:
        print("No YES/NO answers found in any files")
    
    # Create results dictionary
    results = {
        "file_features": file_features,
        "token_to_top_features": {k: v for k, v in token_to_top_features.items()},
        "feature_to_tokens": {int(k): v for k, v in feature_to_tokens.items()},
        "token_analysis_results": token_analysis_results,
        "file_answers": file_answers
    }
    
    return results


def analyze_token_feature_patterns(token_to_top_features):
    """Analyze which features most frequently activate for each token"""
    token_feature_frequency = {}
    
    # For each token, count how often each feature appears
    for token, features in token_to_top_features.items():
        # Create Counter for features
        feature_counts = Counter()
        feature_total_activations = defaultdict(float)
        
        # Count occurrences and sum activations
        for feat_id, activation in features:
            feature_counts[feat_id] += 1
            feature_total_activations[feat_id] += activation
        
        # Calculate average activation for each feature
        feature_avg_activations = {
            feat_id: total_act / count 
            for feat_id, total_act in feature_total_activations.items()
            for count in [feature_counts[feat_id]]
        }
        
        # Store results for this token
        token_feature_frequency[token] = {
            'counts': feature_counts,
            'avg_activations': feature_avg_activations
        }
    
    return token_feature_frequency


def visualize_results(results, output_folder="outputs/analysis", target_tokens=None):
    file_features = results["file_features"]
    token_to_top_features = defaultdict(list, results["token_to_top_features"])
    feature_to_tokens = defaultdict(list, results["feature_to_tokens"])
    token_analysis_results = results.get("token_analysis_results", [])
    file_answers = results.get("file_answers", {})
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Write answer distribution to file
    with open(os.path.join(output_folder, "answer_distribution.txt"), "w") as f:
        f.write("Answer Distribution\n")
        f.write("==================\n\n")
        
        yes_count = sum(1 for answer in file_answers.values() if answer == 'YES')
        no_count = sum(1 for answer in file_answers.values() if answer == 'NO')
        total_files = len(file_answers)
        
        f.write(f"Total files analyzed: {total_files}\n")
        if total_files > 0:
            f.write(f"YES: {yes_count} files ({(yes_count/total_files)*100:.1f}%)\n")
            f.write(f"NO: {no_count} files ({(no_count/total_files)*100:.1f}%)\n")
        else:
            f.write("No YES/NO answers found in any files\n")
        
        # Write individual file answers
        if total_files > 0:
            f.write("\nDetailed Results:\n")
            for file_id, answer in sorted(file_answers.items()):
                f.write(f"{file_id}: {answer}\n")
    
    # Write token statistics to file if we have token analysis results and target tokens
    if token_analysis_results and target_tokens:
        # First aggregate the token analysis results
        aggregated_stats = aggregate_token_analysis(token_analysis_results, target_tokens, token_to_top_features)
        
        # Create a mapping of tokens to files they appear in
        token_to_files = defaultdict(list)
        for result in token_analysis_results:
            file_id = result['file_id']  # Get the original file ID
            for token in target_tokens:
                if result['token_positions'].get(token):
                    token_to_files[token].append(file_id)
        
        # Write token presence in files
        with open(os.path.join(output_folder, "token_file_presence.txt"), "w") as f:
            f.write("Token Presence in Files\n")
            f.write("=====================\n\n")
            
            for token in target_tokens:
                files = token_to_files[token]
                f.write(f"\nToken: '{token}'\n")
                f.write(f"  Present in {len(files)} files:\n")
                for file_id in sorted(files):
                    f.write(f"    {file_id}\n")
        
        # Write token statistics
        with open(os.path.join(output_folder, "token_statistics.txt"), "w") as f:
            f.write("Token Statistics\n")
            f.write("===============\n\n")
            
            total_files = len(token_analysis_results)
            for token in target_tokens:
                stats = aggregated_stats['token_stats'][token]
                f.write(f"\nToken: '{token}'\n")
                f.write(f"  Total occurrences: {stats['total_occurrences']}\n")
                f.write(f"  Present in {stats['files_present']} out of {total_files} files ({(stats['files_present']/total_files)*100:.1f}%)\n")
                
                # Write most common pairs
                if stats['most_common_pairs']:
                    f.write("  Most common token pairs:\n")
                    for other_token, count in stats['most_common_pairs'].most_common(5):
                        f.write(f"    with '{other_token}': {count} times\n")
                
                # Write feature activation information
                if stats['top_features']:
                    f.write("  Top activated features:\n")
                    # Sort features by average activation
                    sorted_features = sorted(
                        stats['top_features'].items(),
                        key=lambda x: (x[1]['avg_activation'] * x[1]['count']),
                        reverse=True
                    )
                    for feat_id, feat_stats in sorted_features[:5]:  # Show top 5 features
                        f.write(f"    Feature {feat_id}: {feat_stats['count']} occurrences, "
                               f"avg activation: {feat_stats['avg_activation']:.3f}\n")
    
    # Analyze the results - most common features across all files
    all_features = Counter()
    for file_id, features in file_features.items():
        for feature_id, activation in features.items():
            all_features[feature_id] += activation

    print("\nTop activated features across all files:")
    if all_features:
        for feature_id, total_activation in all_features.most_common(20):
            print(f"Feature {feature_id}: {total_activation:.3f}")

        # Write to file
        with open(os.path.join(output_folder, "top_features.txt"), "w") as f:
            f.write("Top activated features across all files:\n")
            for feature_id, total_activation in all_features.most_common(50):
                f.write(f"Feature {feature_id}: {total_activation:.3f}\n")

        # Create a list of the most common features
        common_features = [feature_id for feature_id, _ in all_features.most_common(10)]

        print("\nTop tokens for each common feature:")
        with open(os.path.join(output_folder, "feature_tokens.txt"), "w") as f:
            f.write("Top tokens for each common feature:\n")
            
            for feature_id in common_features:
                # Sort tokens by activation value
                tokens = sorted(feature_to_tokens[feature_id], key=lambda x: x[1], reverse=True)
                print(f"\nFeature {feature_id} is most active in:")
                f.write(f"\nFeature {feature_id} is most active in:\n")
                
                for token, activation in tokens[:20]:  # Show top 20 tokens
                    print(f"  Token: '{token}', Activation: {activation:.3f}")
                    f.write(f"  Token: '{token}', Activation: {activation:.3f}\n")
    else:
        print("No features were found in the analysis.")


def analyze_feature_activations(model, tokenizer, inputs, feature_id, sae, layer_id):
    """Analyze activation patterns for a specific feature"""
    results = {
        'activations': [],  # Raw activation values
        'token_activations': [],  # (token, activation) pairs
        'activation_distribution': None,  # Will store histogram data
    }
    
    # Get activations from the model's layer
    activations = gather_residual_activations(model, layer_id, inputs)
    
    # Ensure SAE and activations are on the same device
    sae, activations = ensure_same_device(sae, activations)
    
    # Get the decoded tokens for reference
    decoded_tokens = tokenizer.batch_decode([[token.item()] for token in inputs[0]])
    
    # Encode activations with SAE to get sparse features
    with torch.no_grad():
        if len(activations.shape) == 3:  # [batch, seq_len, hidden_dim]
            features = sae.encode(activations.float())
        else:  # [seq_len, hidden_dim] or other shape
            reshaped_activations = activations.float().unsqueeze(0) if len(activations.shape) == 2 else activations.float()
            features = sae.encode(reshaped_activations)
    
    # Get activations for the specific feature
    feature_activations = features[0, :, feature_id].cpu().numpy()
    results['activations'].extend(feature_activations)
    
    # Record token-activation pairs
    for i, activation in enumerate(feature_activations):
        if i < len(decoded_tokens):
            token = decoded_tokens[i].strip()
            if token:  # Only include non-empty tokens
                results['token_activations'].append((token, float(activation)))
    
    return results


def visualize_feature_analysis(all_results, output_folder="outputs/analysis"):
    """Visualize feature analysis results"""
    os.makedirs(output_folder, exist_ok=True)
    
    for feature_id, results in all_results.items():
        print(f"\nAnalyzing Feature {feature_id}...")
        
        # Aggregate all activations
        all_activations = np.array(results['activations'])
        if len(all_activations) == 0:
            print(f"No activations found for Feature {feature_id}")
            continue
            
        # Create activation distribution plot
        plt.figure(figsize=(12, 8))
        
        # Create histogram
        counts, bins, _ = plt.hist(all_activations, bins=50, density=True, alpha=0.7, 
                                 color='blue', label='Activation Distribution')
        
        # Add density curve
        density = gaussian_kde(all_activations)
        xs = np.linspace(min(bins), max(bins), 200)
        plt.plot(xs, density(xs), 'r-', lw=2, label='Density Curve')
        
        plt.title(f'Activation Distribution (Feature {feature_id})')
        plt.xlabel('Activation Level')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(output_folder, f'feature_{feature_id}_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved distribution plot to {plot_path}")
        
        # Write activation statistics
        stats_path = os.path.join(output_folder, f'feature_{feature_id}_analysis.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Feature {feature_id} Analysis\n")
            f.write("=" * 30 + "\n\n")
            
            # Basic statistics
            f.write("Activation Statistics:\n")
            f.write(f"Mean activation: {np.mean(all_activations):.3f}\n")
            f.write(f"Median activation: {np.median(all_activations):.3f}\n")
            f.write(f"Max activation: {np.max(all_activations):.3f}\n")
            f.write(f"Min activation: {np.min(all_activations):.3f}\n")
            f.write(f"Standard deviation: {np.std(all_activations):.3f}\n\n")
            
            # Token analysis
            token_stats = defaultdict(list)
            for file_token_activations in results['token_activations']:
                for token, activation in file_token_activations:
                    token_stats[token].append(activation)
            
            if token_stats:
                # Calculate average activation for each token
                avg_token_activations = {
                    token: np.mean(activations)
                    for token, activations in token_stats.items()
                }
                
                # Sort tokens by average activation
                sorted_tokens = sorted(avg_token_activations.items(), 
                                    key=lambda x: x[1], reverse=True)
                
                f.write("Top Activated Tokens:\n")
                for token, avg_activation in sorted_tokens[:20]:  # Show top 20 tokens
                    count = len(token_stats[token])
                    f.write(f"  Token: '{token}'\n")
                    f.write(f"    Average activation: {avg_activation:.3f}\n")
                    f.write(f"    Occurrences: {count}\n")
            else:
                f.write("No token activations found for this feature.\n")
        
        print(f"Saved analysis to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze penguin files with SAE')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to analyze')
    parser.add_argument('--device', type=str, default="mps", help='Device to use (mps, cuda, cpu)')
    parser.add_argument('--layer_id', type=int, default=19, help='Layer ID to analyze')
    parser.add_argument('--target_tokens', nargs='+', help='List of specific tokens to analyze')
    parser.add_argument('--target_features', nargs='+', type=int, help='List of specific features to analyze')
    parser.add_argument('--features_only', action='store_true', help='Only run feature analysis without token analysis')
    parser.add_argument('--input_folder', type=str, default="outputs/penguin/raw_outputs", help='Input folder containing penguin .pt files')
    parser.add_argument('--output_folder', type=str, default="outputs/analysis", help='Output folder for analysis results')
    args = parser.parse_args()
    
    print("Loading model and SAE...")
    
    # Load the SAE and model
    sae_name = "DeepSeek-R1-Distill-Llama-8B-SAE-l19"
    file_path = hf_hub_download(
        repo_id=f"qresearch/{sae_name}",
        filename=f"{sae_name}.pt",
        repo_type="model"
    )

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    expansion_factor = 16
    sae = SparseAutoencoder.from_pretrained(
        path=file_path,
        input_dim=model.config.hidden_size,
        expansion_factor=expansion_factor,
        device=args.device
    )
    
    # Run regular token analysis if not features_only
    if not args.features_only:
        results = analyze_all_penguin_files(
            model=model, 
            tokenizer=tokenizer, 
            sae=sae, 
            layer_id=args.layer_id,
            device=args.device,
            max_files=args.max_files,
            target_tokens=args.target_tokens,
            input_folder=args.input_folder,
            output_folder=args.output_folder
        )
        
        # Save results
        os.makedirs(args.output_folder, exist_ok=True)
        with open(os.path.join(args.output_folder, "analysis_results.pkl"), "wb") as f:
            pickle.dump(results, f)
        
        # Visualize results
        visualize_results(results, output_folder=args.output_folder, target_tokens=args.target_tokens)
        
        # Clear memory after token analysis
        del results
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # If target features specified, run feature analysis
    if args.target_features:
        print("\nRunning feature-specific analysis...")
        feature_results = defaultdict(lambda: {
            'activations': [],
            'token_activations': []
        })
        
        # Process each file
        penguin_files = glob.glob(os.path.join(args.input_folder, "*.pt"))
        if args.max_files:
            penguin_files = penguin_files[:args.max_files]
        
        print(f"Analyzing {len(penguin_files)} files for feature patterns...")
        for pt_file_path in tqdm(penguin_files):
            try:
                # Load and process the file
                loaded_tokens = torch.load(pt_file_path, weights_only=True)
                inputs = loaded_tokens.to(args.device)
                
                # Analyze each requested feature
                for feature_id in args.target_features:
                    results = analyze_feature_activations(model, tokenizer, inputs, feature_id, sae, args.layer_id)
                    feature_results[feature_id]['activations'].extend(results['activations'])
                    feature_results[feature_id]['token_activations'].append(results['token_activations'])
                
                # Clear memory after processing each file
                del inputs, loaded_tokens
                if args.device == "mps":
                    torch.mps.empty_cache()
                elif args.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing {pt_file_path}: {str(e)}")
        
        # Create feature-specific output folder
        feature_output_folder = os.path.join(args.output_folder, "feature_analysis")
        os.makedirs(feature_output_folder, exist_ok=True)
        
        # Visualize feature analysis results
        visualize_feature_analysis(feature_results, feature_output_folder)
        print(f"Feature analysis complete! Results saved to {feature_output_folder}/")
        
        # Clear memory after feature analysis
        del feature_results
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nAll analysis complete! Results saved to {args.output_folder}/")


if __name__ == "__main__":
    main() 