import torch
import os
import re
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# Add the custom logits processor from SAE_2B.ipynb
# -------------------------------
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
        return scores

# -------------------------------
# Helper functions from your original file.
# -------------------------------
def find_numbers(x: str) -> list[str]:
    numbers = re.compile(r'-?[\d,]*\.?\d+', re.MULTILINE | re.DOTALL | re.IGNORECASE).findall(x)
    return numbers

def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', x)
    if boxed_match:
        boxed_content = boxed_match.group(1)
        numbers = find_numbers(boxed_content)
        if numbers:
            return numbers[0]
        return boxed_content
        
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ''

def maybe_remove_comma(x: str) -> str:
    return x.replace(',', '')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate GSM8K responses from the testing set')
    parser.add_argument('--start', type=int, required=True, help='Starting index for generation')
    parser.add_argument('--end', type=int, required=True, help='Ending index for generation')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help='Model name or path to use for generation')
    parser.add_argument('--output_folder', type=str, default="outputs", 
                        help='Output folder for storing results')
    args = parser.parse_args()

    process_id = f"Process[{args.start}-{args.end}]"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"{process_id} Using device: {device}")

    # Load model and tokenizer
    model_name = args.model_name
    print(f"{process_id} Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the mapping dictionary for interventions.
    # We use the same conditioning strings and injection string as in SAE_2B.ipynb.
    conditioning_strs = [" wait", " Wait", "wait", "Wait", " but", " But", "but", "But"]
    injection_str = ", that seems right."
    mapping = {}
    for cond_str in conditioning_strs:
        # Use the first token id for each conditioning string.
        cond_token_id = tokenizer.encode(cond_str, add_special_tokens=False)[0]
        injection_tokens_ids = tokenizer.encode(injection_str, add_special_tokens=False)
        mapping[cond_token_id] = injection_tokens_ids

    # Load dataset
    gsm8k = load_dataset("gsm8k", "main")
    gsm8k_test = gsm8k['test']
    
    # Create output directories if they don't exist
    output_base_dir = args.output_folder
    output_dir = f'{output_base_dir}/intervention_gsm8k_test'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/raw_outputs', exist_ok=True)
    os.makedirs(f'{output_dir}/decoded_text', exist_ok=True)
    
    TEMPLATE = "\nQ: {question}\nA:"
    
    correct = 0
    processed = 0
    
    # Process the specified range of examples
    for task_id in range(args.start, min(args.end, len(gsm8k_test))):
        print(f"{process_id} Processing task_id {task_id} ({processed+1}/{args.end-args.start})")
        problem = gsm8k_test[task_id]
        prompt = TEMPLATE.format(question=problem['question'])
        
        # Create the input using the chat template
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        
        # Initialize a new logits processor for this generation call.
        custom_processor = MultiInterventionLogitsProcessor(mapping)
        logits_processor = LogitsProcessorList([custom_processor])
        
        # Generate response with the custom logits processor applied.
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                logits_processor=logits_processor  # <-- Intervention applied here
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_parts = full_response.split(prompt)
        if len(response_parts) > 1:
            response_text = response_parts[1].strip()
        else:
            response_text = full_response
        
        # Save raw outputs and decoded text
        torch.save(outputs, f'{output_dir}/raw_outputs/output_{task_id}.pt')
        with open(f'{output_dir}/decoded_text/text_{task_id}.txt', 'w') as f:
            f.write(full_response)
        
        short_response = maybe_remove_comma(find_number(response_text))
        print(f"{process_id} Short answer: {short_response}")
        
        ground_truth = maybe_remove_comma(find_number(problem['answer']))
        print(f"{process_id} Ground truth: {ground_truth}")
        
        try:
            is_correct = float(short_response) == float(ground_truth)
        except:
            is_correct = short_response == ground_truth
        
        if is_correct:
            correct += 1
        
        print(f"{process_id} Correct: {is_correct}")
        print(f"{process_id} Running accuracy: {correct}/{processed+1} ({(correct/(processed+1))*100:.2f}%)")
        print("-" * 40)
        processed += 1
        
        del outputs, inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()
    
    print(f"{process_id} Final accuracy: {correct}/{processed} ({(correct/processed)*100:.2f}%)")