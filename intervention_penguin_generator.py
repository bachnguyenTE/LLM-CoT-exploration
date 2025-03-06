import os
import torch
import argparse
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LogitsProcessor, LogitsProcessorList
import re

# Suppress warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom logits processor for interventions
class MultiInterventionLogitsProcessor(LogitsProcessor):
    def __init__(self, mapping):
        """
        mapping: dict mapping conditioning token id (int) to a list (sequence) of injection token ids (list of ints)
        For example, if you want that when the model outputs a token corresponding to " wait"
        it injects tokens for ", that seems right." you can specify that here.
        """
        self.mapping = mapping
        # Dictionary to hold per-beam injection queues: beam index -> list of injection token ids left to force
        self.injection_state = {}

    def __call__(self, input_ids, scores):
        # input_ids shape: (batch_size, sequence_length)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # If this beam is in injection mode, force the next token to be the next in the sequence.
            if i in self.injection_state and self.injection_state[i]:
                next_injection = self.injection_state[i].pop(0)
                scores[i, :] = -float('inf')
                scores[i, next_injection] = 0.0
                if not self.injection_state[i]:
                    del self.injection_state[i]
                continue

            # Otherwise, check if the last generated token is one of our conditioning tokens.
            last_token = input_ids[i, -1].item()
            if last_token in self.mapping:
                # Begin the injection sequence for this beam.
                self.injection_state[i] = self.mapping[last_token].copy()
                next_injection = self.injection_state[i].pop(0)
                scores[i, :] = -float('inf')
                scores[i, next_injection] = 0.0
                if not self.injection_state[i] and i in self.injection_state:
                    del self.injection_state[i]
        return scores

def find_numbers(x: str) -> list[str]:
    numbers = re.compile(r'-?[\d,]*\.?\d+', re.MULTILINE | re.DOTALL | re.IGNORECASE).findall(x)
    return numbers

def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(',', '')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate penguin responses with interventions')
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
    
    # Set device appropriately
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"{process_id} Using device: {device}")
    
    model_name = args.model_name
    print(f"{process_id} Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    AutoConfig.from_pretrained(model_name)

    # Create output directories if they don't exist
    output_base_dir = args.output_folder
    os.makedirs(f'{output_base_dir}/penguin_intervention', exist_ok=True)
    os.makedirs(f'{output_base_dir}/penguin_intervention/raw_outputs', exist_ok=True)
    os.makedirs(f'{output_base_dir}/penguin_intervention/decoded_text', exist_ok=True)

    # Create the mapping dictionary for interventions.
    # We use the same conditioning strings and injection string as in the GSM8K implementation.
    conditioning_strs = [" wait", " Wait", "wait", "Wait", " but", " But", "but", "But"]
    injection_str = ", that seems right."
    mapping = {}
    for cond_str in conditioning_strs:
        # Use the first token id for each conditioning string.
        cond_token_id = tokenizer.encode(cond_str, add_special_tokens=False)[0]
        injection_tokens_ids = tokenizer.encode(injection_str, add_special_tokens=False)
        mapping[cond_token_id] = injection_tokens_ids
    
    # Initialize a new logits processor for this generation call.
    custom_processor = MultiInterventionLogitsProcessor(mapping)
    logits_processor = LogitsProcessorList([custom_processor])

    # Convert temperature to string with underscore instead of decimal
    temp_str = str(args.temperature).replace('.', '_')

    for i in range(args.start, args.end):
        prompt = "Can penguins fly? Segment the thinking process into clear steps and indicate \"YES\" or \"NO\" once at the end."
        
        print(f"{process_id} Processing example {i}")
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ],
            return_tensors="pt"
        ).to(model.device)

        # Generate with interventions
        generation_kwargs = {
            "max_new_tokens": 1200,
            "temperature": args.temperature,
            "logits_processor": logits_processor  # Always apply the intervention
        }
            
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                **generation_kwargs
            )
        
        # Process the output
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        decoded_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Save the outputs
        raw_output_file = f"{output_base_dir}/penguin_intervention/raw_outputs/example_{i}_temp_{temp_str}.txt"
        decoded_file = f"{output_base_dir}/penguin_intervention/decoded_text/example_{i}_temp_{temp_str}.txt"
        
        with open(raw_output_file, "w") as f:
            f.write(raw_output)
        
        with open(decoded_file, "w") as f:
            f.write(decoded_text)
            
        print(f"{process_id} Completed example {i}")
    
    print(f"{process_id} Completed all examples from {args.start} to {args.end-1}")