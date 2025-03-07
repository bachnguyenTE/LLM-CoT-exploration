import os
import re
import numpy as np
from datasets import load_dataset
import argparse

def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    numbers = re.compile(r'-?[\d,]*\.?\d+').findall(x)
    return numbers

def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
    """
    Extracts number using the same method as GSM8K_generate_from_test.py.
    First looks for text after "The answer is" if present,
    otherwise uses the last number in the text.
    """
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

def find_number_outside_think(x: str) -> str:
    """
    Finds the most relevant number in a string outside of <think> tags.
    Uses boxed format if available, otherwise uses find_number approach.
    """
    # Remove text within <think>...</think>
    outside_think = re.sub(r'<think>.*?</think>', '', x, flags=re.DOTALL)
    
    # Check for \boxed{} outside of <think> tags
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', outside_think)
    if boxed_match:
        boxed_content = boxed_match.group(1)
        numbers = find_numbers(boxed_content)
        if numbers:
            return numbers[0]
        return boxed_content

    # If no boxed answer, use the standard find_number approach
    return find_number(outside_think)

def maybe_remove_comma(x: str) -> str:
    return x.replace(',', '')

def extract_question(text: str) -> str:
    """Extract the question from a model response."""
    # Look for the format: 
    # Q: [actual question]
    match = re.search(r'Q:\s*(.*?)(?=<|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Backup method: Just get the first few sentences after Q:
    match = re.search(r'Q:\s*(.*?)(?=\n\n|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

def load_responses(directory):
    responses = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            task_id = int(filename.split('_')[-1].split('.')[0])
            with open(os.path.join(directory, filename), 'r') as f:
                responses[task_id] = f.read()
    return responses

def create_question_mapping(gsm8k_data, response_dir):
    """Create a mapping between response file IDs and GSM8K dataset indices."""
    responses = load_responses(response_dir)
    
    # Build a simple question-to-index mapping from GSM8K data
    gsm8k_question_to_idx = {}
    for i, item in enumerate(gsm8k_data):
        # Get first 50 chars of question for matching
        question_start = item['question'][:100].lower().strip()
        gsm8k_question_to_idx[question_start] = i
        
    # Create mapping from response IDs to GSM8K indices
    id_mapping = {}
    not_found = []
    
    for resp_id, response_text in responses.items():
        question = extract_question(response_text)
        if not question:
            continue
            
        # Get first 50 chars for matching
        question_start = question[:100].lower().strip()
        
        # Look for a match in the GSM8K dataset
        found = False
        for gsm_question_start, gsm_idx in gsm8k_question_to_idx.items():
            if question_start in gsm_question_start or gsm_question_start in question_start:
                id_mapping[resp_id] = gsm_idx
                found = True
                break
                
        if not found:
            not_found.append(resp_id)
            
    return id_mapping, not_found

def compare_responses(normal_dir, intervention_dir, gsm8k_data, id_mapping, max_questions=None):
    normal_responses = load_responses(normal_dir)
    intervention_responses = load_responses(intervention_dir)

    normal_wrong_intervention_right = []
    intervention_wrong_normal_right = []
    normal_undefined_intervention_right = []
    intervention_undefined_normal_right = []
    both_undefined = []
    both_wrong_count = 0
    both_right_count = 0
    
    # For tracking standard vs. relaxed matching
    normal_boxed_wrong_but_find_number_right = []
    intervention_boxed_wrong_but_find_number_right = []
    
    total_questions = 0
    
    # Use the same set of question IDs for which we have a mapping
    question_ids = list(id_mapping.keys())
    
    # Limit the number of questions if specified
    if max_questions and max_questions < len(question_ids):
        question_ids = question_ids[:max_questions]
    
    for resp_id in question_ids:
        # Get the corresponding GSM8K index
        gsm_idx = id_mapping[resp_id]
        
        # Get ground truth answer
        ground_truth = gsm8k_data[gsm_idx]['answer']
        
        normal_response = normal_responses.get(resp_id, '')
        intervention_response = intervention_responses.get(resp_id, '')
        
        # Skip if either response is missing
        if not normal_response or not intervention_response:
            continue
            
        total_questions += 1

        # Check if the answer is undefined (no \boxed{} found outside <think> tags)
        normal_undefined = '\\boxed{' not in re.sub(r'<think>.*?</think>', '', normal_response, flags=re.DOTALL)
        intervention_undefined = '\\boxed{' not in re.sub(r'<think>.*?</think>', '', intervention_response, flags=re.DOTALL)

        # Extract answers using the strict method (boxed preferred)
        normal_answer_strict = ''
        if not normal_undefined:
            outside_think = re.sub(r'<think>.*?</think>', '', normal_response, flags=re.DOTALL)
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', outside_think)
            if boxed_match:
                boxed_content = boxed_match.group(1)
                numbers = find_numbers(boxed_content)
                if numbers:
                    normal_answer_strict = maybe_remove_comma(numbers[0])
                else:
                    normal_answer_strict = maybe_remove_comma(boxed_content)
        
        intervention_answer_strict = ''
        if not intervention_undefined:
            outside_think = re.sub(r'<think>.*?</think>', '', intervention_response, flags=re.DOTALL)
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', outside_think)
            if boxed_match:
                boxed_content = boxed_match.group(1)
                numbers = find_numbers(boxed_content)
                if numbers:
                    intervention_answer_strict = maybe_remove_comma(numbers[0])
                else:
                    intervention_answer_strict = maybe_remove_comma(boxed_content)
        
        # Extract answers using the relaxed method (same as GSM8K_generate_from_test.py)
        normal_answer_relaxed = maybe_remove_comma(find_number(re.sub(r'<think>.*?</think>', '', normal_response, flags=re.DOTALL)))
        intervention_answer_relaxed = maybe_remove_comma(find_number(re.sub(r'<think>.*?</think>', '', intervention_response, flags=re.DOTALL)))
        
        # Extract ground truth using the same relaxed method
        ground_truth_answer = maybe_remove_comma(find_number(ground_truth))
        
        # Track cases where boxed is wrong but relaxed method is right
        if normal_answer_strict and normal_answer_strict != ground_truth_answer and normal_answer_relaxed == ground_truth_answer:
            normal_boxed_wrong_but_find_number_right.append(resp_id)
            
        if intervention_answer_strict and intervention_answer_strict != ground_truth_answer and intervention_answer_relaxed == ground_truth_answer:
            intervention_boxed_wrong_but_find_number_right.append(resp_id)
        
        # Use relaxed answers for all comparisons to match GSM8K_generate_from_test.py
        normal_correct = normal_answer_relaxed == ground_truth_answer
        intervention_correct = intervention_answer_relaxed == ground_truth_answer

        if normal_correct and intervention_correct:
            both_right_count += 1
        elif normal_undefined and intervention_undefined:
            both_undefined.append(resp_id)
        elif not normal_correct and not intervention_correct:
            both_wrong_count += 1
        elif normal_undefined and intervention_correct:
            normal_undefined_intervention_right.append(resp_id)
        elif intervention_undefined and normal_correct:
            intervention_undefined_normal_right.append(resp_id)
        elif not normal_correct and intervention_correct:
            normal_wrong_intervention_right.append(resp_id)
        elif normal_correct and not intervention_correct:
            intervention_wrong_normal_right.append(resp_id)

    return (normal_wrong_intervention_right, intervention_wrong_normal_right, 
            normal_undefined_intervention_right, intervention_undefined_normal_right, 
            both_undefined, both_wrong_count, both_right_count, total_questions,
            normal_boxed_wrong_but_find_number_right, intervention_boxed_wrong_but_find_number_right)

def save_results_to_files(results, output_dir):
    """Save results to text files and numpy arrays."""
    os.makedirs(output_dir, exist_ok=True)
    
    categories = [
        "normal_wrong_intervention_right",
        "intervention_wrong_normal_right",
        "normal_undefined_intervention_right",
        "intervention_undefined_normal_right",
        "both_undefined"
    ]
    
    # Save each category to a text file
    for category, data in zip(categories, results):
        with open(os.path.join(output_dir, f"{category}.txt"), 'w') as f:
            for item in data:
                f.write(f"{item}\n")
    
    # Save all categories as a single numpy dictionary
    np_dict = {category: np.array(data) for category, data in zip(categories, results)}
    np.save(os.path.join(output_dir, "results_dict.npy"), np_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare normal and intervention answers for GSM8K dataset.")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test",
                        help="Dataset split to use: train or test (used only if normal_dir and intervention_dir not specified)")
    parser.add_argument("--normal_dir", type=str, default=None,
                        help="Directory containing normal responses (overrides split-based default)")
    parser.add_argument("--intervention_dir", type=str, default=None,
                        help="Directory containing intervention responses (overrides split-based default)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save comparison results (defaults to meta_analysis directory in parent of normal_dir)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to process (default: process all available questions)")
    args = parser.parse_args()
    
    # Load GSM8K dataset
    gsm8k = load_dataset("gsm8k", "main")
    split = args.split
    if split == "train":
        gsm8k_split = gsm8k['train']
    else:
        gsm8k_split = gsm8k['test']
    
    # Set directories based on command line arguments or defaults based on split
    if args.normal_dir and args.intervention_dir:
        normal_dir = args.normal_dir
        intervention_dir = args.intervention_dir
        
        # Determine output directory if not specified
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Determine parent directory of normal_dir and create meta_analysis subdirectory
            parent_dir = os.path.dirname(os.path.abspath(normal_dir))
            base_name = os.path.basename(os.path.normpath(intervention_dir))
            output_dir = os.path.join(parent_dir, "meta_analysis", base_name)
    else:
        # Use default directories based on split
        if split == "train":
            normal_dir = 'outputs/gsm8k_train/decoded_text'
            intervention_dir = 'outputs/intervention_gsm8k_train/decoded_text'
            output_dir = 'outputs/meta_analysis/intervention_gsm8k_train'
        else:
            normal_dir = 'outputs/gsm8k_test/decoded_text'
            intervention_dir = 'outputs/intervention_gsm8k_test/decoded_text'
            output_dir = 'outputs/meta_analysis/intervention_gsm8k_test'

    print(f"Comparing responses from:")
    print(f"  Normal directory: {normal_dir}")
    print(f"  Intervention directory: {intervention_dir}")
    print(f"Saving results to: {output_dir}")
    if args.max_questions:
        print(f"Processing up to {args.max_questions} questions")
        
    # Create mapping between response IDs and GSM8K dataset indices
    print("Creating mapping between response files and GSM8K dataset...")
    id_mapping, not_found = create_question_mapping(gsm8k_split, normal_dir)
    print(f"Found mappings for {len(id_mapping)} questions")
    print(f"Could not find mappings for {len(not_found)} questions")
    
    # Compare responses with the corrected mapping
    results = compare_responses(normal_dir, intervention_dir, gsm8k_split, id_mapping, args.max_questions)

    # Output results
    def print_results(description, count, total):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{description}: {count} out of {total} ({percentage:.2f}%)")

    print("\n--- Standard Comparison Results ---")
    print_results("Questions normal got wrong but intervention got right", len(results[0]), results[-3])
    print_results("Questions intervention got wrong but normal got right", len(results[1]), results[-3])
    print_results("Questions normal was undefined but intervention got right", len(results[2]), results[-3])
    print_results("Questions intervention was undefined but normal got right", len(results[3]), results[-3])
    print_results("Questions both were undefined", len(results[4]), results[-3])
    print_results("Number of questions both got wrong", results[5], results[-3])
    print_results("Number of questions both got right", results[6], results[-3])
    
    # Print additional information about answer matching methods
    print("\n--- Answer Extraction Method Comparison ---")
    print_results("Normal: boxed wrong but find_number correct", len(results[8]), results[-3])
    print_results("Intervention: boxed wrong but find_number correct", len(results[9]), results[-3])

    # Calculate the number of questions each got right
    normal_right = results[6] + len(results[1]) + len(results[3])
    intervention_right = results[6] + len(results[0]) + len(results[2])

    # Print overall accuracy matching GSM8K_generate_from_test.py method
    print("\n--- Overall Accuracy (Using GSM8K_generate_from_test.py extraction method) ---")
    print_results("Normal accuracy", normal_right, results[-3])
    print_results("Intervention accuracy", intervention_right, results[-3])

    # Calculate the number of questions each got right excluding undefined
    # Include cases where the other model was undefined but this one was right
    normal_right_excluding_undefined = results[6] + len(results[1]) + len(results[3])
    intervention_right_excluding_undefined = results[6] + len(results[0]) + len(results[2])

    # Calculate the total number of non-undefined questions for each set
    total_non_undefined_normal = results[-3] - len(results[4]) - len(results[2])
    total_non_undefined_intervention = results[-3] - len(results[4]) - len(results[3])

    # Print the adjusted results with correct percentages
    print("\n--- Accuracy Excluding Undefined Questions ---")
    print_results("Number of questions normal got right excluding undefined", normal_right_excluding_undefined, total_non_undefined_normal)
    print_results("Number of questions intervention got right excluding undefined", intervention_right_excluding_undefined, total_non_undefined_intervention)

    # Save results to files
    save_results_to_files(results[:5], output_dir)  # Only save the lists, not the counts 