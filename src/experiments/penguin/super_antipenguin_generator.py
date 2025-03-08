import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import argparse
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import gc
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate penguin responses with specified index range')
    parser.add_argument('--start', type=int, required=True, help='Starting index for generation')
    parser.add_argument('--end', type=int, required=True, help='Ending index for generation')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation')
    
    args = parser.parse_args()
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    AutoConfig.from_pretrained(model_name)

    # Create output directories if they don't exist
    os.makedirs('outputs/super_antipenguin', exist_ok=True)
    os.makedirs('outputs/super_antipenguin/raw_outputs', exist_ok=True)
    os.makedirs('outputs/super_antipenguin/decoded_text', exist_ok=True)

    # Convert temperature to string with underscore instead of decimal
    temp_str = str(args.temperature).replace('.', '_')

    for i in range(args.start, args.end):
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Can penguins fly? Segment the thinking process into clear steps and indicate \"YES\" or \"NO\" once at the end ."},
                {"role": "assistant", "content": "I must not use \"Wait\" or \"wait\" in my thinking process."}
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("mps")

        outputs = None
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=1200, temperature=args.temperature)
        decoded_text = tokenizer.decode(outputs[0])
        
        # Save raw outputs
        torch.save(outputs, f'outputs/super_antipenguin/raw_outputs/output_{i}_temp{temp_str}.pt')
        
        # Save decoded text
        with open(f'outputs/super_antipenguin/decoded_text/text_{i}_temp{temp_str}.txt', 'w') as f:
            f.write(decoded_text)

        # Clear memory
        del outputs
        del inputs
        del decoded_text
        torch.mps.empty_cache()
        gc.collect()
