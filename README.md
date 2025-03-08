# LLM Chain-of-Thought Exploration

This repository contains tools and experiments for exploring Large Language Model chain-of-thought reasoning, focusing on how models approach reasoning tasks and the impact of different prompting strategies.

## Repository Structure

```
.
├── src/                           # Main source code directory
│   ├── experiments/               # Experimental code for different datasets
│   │   ├── gsm8k/                 # GSM8K math reasoning experiments
│   │   └── penguin/               # Penguin classification experiments
│   ├── utils/                     # Utility functions used across experiments
│   ├── models/                    # Model-related code and interfaces
│   ├── analysis/                  # Analysis scripts for experimental results
│   ├── visualization/             # Visualization tools for results
│   └── scripts/                   # Scripts for maintaining the codebase
├── activation_extraction/         # Package for extracting and analyzing model activations
├── outputs_small/                 # Raw model outputs for various datasets and tasks
├── outputs/                       # Full output data from experiments
├── activations/                   # Extracted model activations organized by model and dataset
├── viz_output/                    # Visualizations of model activations
└── final_plots/                   # Final plots and figures for analysis
```

## Experiments Overview

This project explores various aspects of chain-of-thought reasoning in LLMs through multiple experiments:

### 1. GSM8K Math Reasoning

Evaluates how models approach mathematical reasoning tasks from the GSM8K dataset, with various experimental conditions:

- **Standard CoT**: Models generate step-by-step reasoning before arriving at an answer
- **Anti-CoT**: Models are instructed to provide contrary or incorrect reasoning
- **Unthink**: Models are asked to generate answers without showing reasoning steps
- **Intervention**: Specific interventions are made during the reasoning process

Scripts are located in `src/experiments/gsm8k/`.

### 2. Penguin Classification

Explores a simple binary classification task asking "Can penguins fly?" to test model knowledge and reasoning:

- **Standard CoT**: Models explain their thinking process before answering
- **Anti-CoT**: Models are encouraged to provide contrary reasoning
- **Super Anti-CoT**: Enhanced contrary reasoning experiment
- **Unthink**: Models answer without showing reasoning
- **Intervention**: Specific interventions during the reasoning process

Scripts are located in `src/experiments/penguin/`.

## Activation Extraction Tools

The `activation_extraction` package provides tools for extracting, analyzing, and visualizing model activations:

- Extract attention patterns, hidden states, and logits from transformer models
- Process single files or batch directories
- Verify activation file integrity
- Visualize attention patterns, hidden state PCA projections, and token predictions

For more details, see the [Activation Extraction README](activation_extraction/README.md).

## Analysis Tools

The repository includes various analysis tools:

- `src/analysis/run_analysis.py`: General analysis script
- `src/analysis/compare_answers_GSM8K.py`: Compares different model answers on GSM8K tasks
- `src/analysis/compare_intervention_GSM8K.py`: Analyzes intervention results on GSM8K tasks

## Sample Generated Data

Text and token data generated from Deepseek R1 Distill Qwen-1.5B are available in `outputs_small`.

## How to Run Experiments

### Setting Up

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LLM-CoT-exploration.git
   cd LLM-CoT-exploration
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running GSM8K Experiments

```bash
# Using Python script directly
cd src/experiments/gsm8k
python GSM8K_generate_from_test.py --start 0 --end 10 --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Using shell script
cd src/experiments/gsm8k
./run_GSM8K_test.sh 0 10 1 "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
```

### Running Penguin Classification Experiments

```bash
# Using Python script directly
cd src/experiments/penguin
python penguin_generator.py --start 0 --end 10 --temperature 0.6

# Using shell script
cd src/experiments/penguin
./run_penguins.sh 0 10 1 0.6
```

### Extracting Model Activations

```bash
cd src/utils
python run_activations.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
                          --input_dir "outputs_small/gsm8k_test/raw_outputs" \
                          --dataset "gsm8k_test" \
                          --num_files 5
```

### Analyzing Results

```bash
cd src/analysis
python run_analysis.py
```

## Requirements

Main dependencies include:

- PyTorch
- Transformers
- Matplotlib
- Seaborn
- Scikit-learn
- TQDM
- Pandas
- NumPy

For a full list, see `requirements.txt`.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Contributors

This project is maintained by researchers exploring LLM chain-of-thought reasoning mechanisms. 
