# LLM Chain-of-Thought Exploration

This repository contains tools and experiments for exploring Large Language Model chain-of-thought reasoning.

## Repository Structure

- **activation_extraction/**: Package for extracting and analyzing model activations
- **outputs_small/**: Raw model outputs for various datasets and tasks
- **activations/**: Extracted model activations organized by model and dataset
- **viz_output/**: Visualizations of model activations

## Activation Extraction Tools

The `activation_extraction` package provides a comprehensive set of tools for extracting, analyzing, and visualizing model activations. This is useful for research into model interpretability, understanding attention patterns, and analyzing hidden state representations.

### Key Features

- Extract attention activations, hidden states, and logits from transformer models
- Process single files or batch directories
- Verify activation file integrity
- Visualize attention patterns, hidden state PCA projections, and token predictions

### How to Use

```bash
# Run activation extraction
./run_activations.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
                    --input_dir "outputs_small/gsm8k_test/raw_outputs" \
                    --dataset "gsm8k_test" \
                    --num_files 5

# Analyze activations programmatically
./use_activations.py
```

For more details, see the [Activation Extraction README](activation_extraction/README.md).

## Penguin Classification Analysis

This repository also contains tools to analyze the responses from a language model answering the question "Can penguins fly?". The correct answer is "NO."

### Penguin Analysis Scripts

- **penguin_generator.py**: Generates model responses to the question
- **parse_penguin_results.py**: Analyzes outputs and counts correct answers
- **visualize_results.py**: Creates visualizations of the classification results
- **run_analysis.py**: Combines parsing and visualization steps
- **reindex_files.py**: Reindexes files to have sequential indices

## Requirements

The main dependencies are:

```
torch
transformers
matplotlib
seaborn
scikit-learn
tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

# LLM-CoT-exploration
 
# Penguin Classification Analysis

This repository contains tools to analyze the responses from a language model answering the question "Can penguins fly?". The correct answer is "NO."

## Scripts

### 1. penguin_generator.py

This script generates model responses to the question "Can penguins fly?" and saves them to files.

Usage:
```bash
python penguin_generator.py --start 0 --end 10 --temperature 0.6
```

Parameters:
- `--start`: Starting index for generation
- `--end`: Ending index for generation
- `--temperature`: Temperature for generation (higher values = more randomness)

### 2. parse_penguin_results.py

This script analyzes the model outputs and counts how many times the model answered correctly.

Usage:
```bash
python parse_penguin_results.py
```

Output:
- Lists each file and its classification (YES, NO, or UNKNOWN)
- Provides a summary of overall accuracy
- Breaks down results by temperature setting

### 3. visualize_results.py

This script creates visualizations of the classification results.

Usage:
```bash
python visualize_results.py
```

Output:
- Creates a bar chart showing classification accuracy by temperature
- Creates a pie chart showing overall classification results
- Saves visualizations to the `outputs/visualizations/` directory

### 4. run_analysis.py

This script combines the parsing and visualization steps into a single command.

Usage:
```bash
python run_analysis.py [--parse-only] [--visualize-only]
```

Parameters:
- `--parse-only`: Only parse results without creating visualizations
- `--visualize-only`: Only create visualizations without parsing results again

Output:
- Runs parsing and visualization (or just one if specified)
- Displays summary information
- Saves visualizations to the `outputs/visualizations/` directory

### 5. reindex_files.py

This script reindexes files from source directories to have sequential indices without gaps. It matches text files (`.txt`) with their corresponding raw output files (`.pt`) based on indices.

Usage:
```bash
python reindex_files.py [options]
```

Parameters:
- `--text-dir`: Directory containing text files (default: outputs/penguin/decoded_text_reindexed)
- `--raw-dir`: Directory containing raw output files (default: outputs/penguin/raw_outputs_reindexed)
- `--target-text-dir`: Target directory for reindexed text files (default: same as source)
- `--target-raw-dir`: Target directory for reindexed raw files (default: same as source)
- `--preserve`: Preserve original files by copying instead of renaming
- `--no-group-by-temp`: Do not group files by temperature when reindexing (by default, files are grouped by temperature)
- `--dry-run`: Print what would be done without making changes
- `--verbose`: Print more detailed information

Output:
- Reindexes files to have sequential indices
- Groups files by temperature to ensure each temperature set has its own sequence starting from 0
- Can copy files to new directories or rename in place
- Shows a summary of reindexing operations

Example:
```bash
# Dry run to see what changes would be made
python reindex_files.py --dry-run --verbose

# Reindex files in place
python reindex_files.py

# Reindex files to new directories
python reindex_files.py --target-text-dir outputs/penguin/decoded_text_sequential --target-raw-dir outputs/penguin/raw_outputs_sequential

# Preserve original files
python reindex_files.py --preserve

# Reindex all files together regardless of temperature
python reindex_files.py --no-group-by-temp
```

## Directory Structure

```
.
├── penguin_generator.py          # Script to generate model responses
├── parse_penguin_results.py      # Script to analyze model outputs
├── visualize_results.py          # Script to create visualizations
├── run_analysis.py               # Script to run analysis
├── reindex_files.py              # Script to reindex files to sequential indices
├── outputs/
│   ├── penguin/
│   │   ├── raw_outputs/          # Raw model outputs (tensor format)
│   │   └── decoded_text/         # Decoded text outputs
│   └── visualizations/           # Generated visualizations
└── README.md                     # This file
```

## How It Works

1. The `penguin_generator.py` script generates responses using a language model.
2. The model outputs include a thinking process and a final classification (YES or NO).
3. `parse_penguin_results.py` extracts the final classification from each file.
4. The script counts how many times the model answered correctly (NO) versus incorrectly (YES).
5. `visualize_results.py` creates visualizations to help analyze patterns in the results.

## Example

Here's an example of a model output:
```
Can penguins fly? Segment the thinking process into clear steps and indicate "YES" or "NO" once at the end .

[Thinking process here...]

So, putting it all together: penguins are birds, but their wings are adapted for swimming. They don't have the structural or anatomical features needed for flight. They move through the water using their flippers, not their wings. Therefore, penguins cannot fly.

NO

Penguins are birds with adapted wings for swimming, not flying. They cannot generate the necessary lift in air and are not observed flying. Their movement through water is facilitated by their flippers, not wings. Therefore, penguins cannot fly.
```

The correct answer is "NO" since penguins cannot fly, and the script will count this as a correct classification.

## How to Run

1. First, generate the model outputs (if not already done):
   ```bash
   python penguin_generator.py --start 0 --end 10 --temperature 0.6
   ```

2. Run the complete analysis with a single command:
   ```bash
   python run_analysis.py
   ```

3. View the visualizations in the `outputs/visualizations/` directory.

Alternatively, you can run each step separately:
```bash
python parse_penguin_results.py  # Parse the results
python visualize_results.py      # Create visualizations
```
 
