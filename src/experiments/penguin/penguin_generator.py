import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import argparse
import sys
import os
import gc
import warnings
import numpy as np

# Add path to the src directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate penguin responses with specified index range')
    parser.add_argument('--start', type=int, required=True, help='Starting index for generation')
    parser.add_argument('--end', type=int, required=True, help='Ending index for generation')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help='Model name or path to use for generation')
    parser.add_argument('--output_folder', type=str, default="outputs", 
                        help='Output folder for storing results')
    
    args = parser.parse_args()
    
    # Process identifier for logging
    process_id = f"Process[{args.start}-{args.end}]"
    
    model_name = args.model_name
    print(f"{process_id} Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    AutoConfig.from_pretrained(model_name)

    # Create output directories if they don't exist
    output_base_dir = args.output_folder
    os.makedirs(f'{output_base_dir}/penguin', exist_ok=True)
    os.makedirs(f'{output_base_dir}/penguin/raw_outputs', exist_ok=True)
    os.makedirs(f'{output_base_dir}/penguin/decoded_text', exist_ok=True)

    # Convert temperature to string with underscore instead of decimal
    temp_str = str(args.temperature).replace('.', '_')

    for i in range(args.start, args.end):
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Can penguins fly? Segment the thinking process into clear steps and indicate \"YES\" or \"NO\" once at the end ."},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("mps")

        outputs = None
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=1200, temperature=args.temperature)
        decoded_text = tokenizer.decode(outputs[0])
        
        # Save raw outputs
        torch.save(outputs, f'{output_base_dir}/penguin/raw_outputs/output_{i}_temp{temp_str}.pt')
        
        # Save decoded text
        with open(f'{output_base_dir}/penguin/decoded_text/text_{i}_temp{temp_str}.txt', 'w') as f:
            f.write(decoded_text)

        # Clear memory
        del outputs
        del inputs
        del decoded_text
        torch.mps.empty_cache()
        gc.collect()
