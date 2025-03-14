import torch
import os
import re
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    # Search for number, possibly negative (hyphen), with thousand separators
    # (comma), and with a decimal point (period inbetween digits).
    numbers = re.compile(
        r'-?[\d,]*\.?\d+',
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers

def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
    """Finds the most relevant number in a string."""
    # If model uses boxed format, extract the number from \boxed{ANSWER}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', x)
    if boxed_match:
        boxed_content = boxed_match.group(1)
        numbers = find_numbers(boxed_content)
        if numbers:
            return numbers[0]
        return boxed_content  # Return the content even if it's not a number
        
    # If model uses the answer delimiter, then select the first number following
    # that format.
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    # In general, select the last number in the string.
    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ''


def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
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
    
    # Create a process identifier for logging purposes
    process_id = f"Process[{args.start}-{args.end}]"
    
    # Set device to MPS (Metal Performance Shaders) for Mac with Apple Silicon, or CUDA/CPU as fallback
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"{process_id} Using device: {device}")
    
    # Load model
    model_name = args.model_name
    print(f"{process_id} Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset using default cache location
    gsm8k = load_dataset("gsm8k", "main")
    gsm8k_test = gsm8k['test']
    
    # Create output directories if they don't exist
    output_base_dir = args.output_folder
    output_dir = f'{output_base_dir}/gsm8k_test'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/raw_outputs', exist_ok=True)
    os.makedirs(f'{output_dir}/decoded_text', exist_ok=True)
    
    # Template for prompts
    TEMPLATE = """
Q: {question}
A:"""
    
    # Track metrics
    correct = 0
    processed = 0
    
    # Process the specified range of examples
    for task_id in range(args.start, min(args.end, len(gsm8k_test))):
        print(f"{process_id} Processing task_id {task_id} ({processed+1}/{args.end-args.start})")
        
        problem = gsm8k_test[task_id]
        
        # Formulate the prompt
        prompt = TEMPLATE.format(question=problem['question'])
        
        # Generate response using the model with chat template
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt}
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        
        # Generate response - using the same parameters as in the notebook
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response (after the prompt)
        response_parts = full_response.split(prompt)
        if len(response_parts) > 1:
            response_text = response_parts[1].strip()
        else:
            response_text = full_response
        
        # Save raw outputs
        torch.save(outputs, f'{output_dir}/raw_outputs/output_{task_id}.pt')
        
        # Save decoded text
        with open(f'{output_dir}/decoded_text/text_{task_id}.txt', 'w') as f:
            f.write(full_response)
        
        # Extract short answer and check if correct
        short_response = maybe_remove_comma(find_number(response_text))
        print(f"{process_id} Short answer: {short_response}")
        
        # Get ground truth
        ground_truth = maybe_remove_comma(find_number(problem['answer']))
        print(f"{process_id} Ground truth: {ground_truth}")
        
        # Check correctness
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
        
        # Clean up to save memory
        del outputs
        del inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()
    
    # Final report
    print(f"{process_id} Final accuracy: {correct}/{processed} ({(correct/processed)*100:.2f}%)") 