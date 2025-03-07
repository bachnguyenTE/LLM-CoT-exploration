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

def count_think_words(response: str) -> int:
    """
    Counts the number of words between the first <think> and the last </think> in the response.
    If either tag is missing or malformed, returns 0.
    """
    start = response.find('<think>')
    end = response.rfind('</think>')
    if start == -1 or end == -1 or end < start:
        return 0
    # Extract text between the first <think> and the last </think>
    content = response[start + len('<think>'):end]
    return len(content.split())

def create_question_mapping(gsm8k_data, response_dir):
    """Create a mapping between response file IDs and GSM8K dataset indices."""
    responses = load_responses(response_dir)
    
    # Build a simple question-to-index mapping from GSM8K data
    gsm8k_question_to_idx = {}
    for i, item in enumerate(gsm8k_data):
        # Get first 100 chars of question for matching
        question_start = item['question'][:100].lower().strip()
        gsm8k_question_to_idx[question_start] = i
        
    # Create mapping from response IDs to GSM8K indices
    id_mapping = {}
    not_found = []
    
    for resp_id, response_text in responses.items():
        question = extract_question(response_text)
        if not question:
            continue
            
        # Get first 100 chars for matching
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

def compare_responses(normal_dir, unthink_dir, gsm8k_data, id_mapping, max_questions=None):
    normal_responses = load_responses(normal_dir)
    unthink_responses = load_responses(unthink_dir)

    normal_wrong_unthink_right = []
    unthink_wrong_normal_right = []
    normal_undefined_unthink_right = []
    unthink_undefined_normal_right = []
    both_undefined = []
    both_wrong_count = 0
    both_right_count = 0
    total_questions = 0
    filtered_ids = []  # To keep track of questions excluded due to unthink's chain-of-thought length
    
    # For tracking standard vs. relaxed matching
    normal_boxed_wrong_but_find_number_right = []
    unthink_boxed_wrong_but_find_number_right = []
    
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
        unthink_response = unthink_responses.get(resp_id, '')
        
        # Skip if either response is missing
        if not normal_response or not unthink_response:
            continue
        
        # Exclude question if there are more than 5 words in the chain-of-thought block of the unthink response.
        if count_think_words(unthink_response) > 5:
            filtered_ids.append(resp_id)
            continue

        total_questions += 1

        # Check if the answer is undefined (no \boxed{} found outside <think> tags)
        normal_undefined = '\\boxed{' not in re.sub(r'<think>.*?</think>', '', normal_response, flags=re.DOTALL)
        unthink_undefined = '\\boxed{' not in re.sub(r'<think>.*?</think>', '', unthink_response, flags=re.DOTALL)

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
        
        unthink_answer_strict = ''
        if not unthink_undefined:
            outside_think = re.sub(r'<think>.*?</think>', '', unthink_response, flags=re.DOTALL)
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', outside_think)
            if boxed_match:
                boxed_content = boxed_match.group(1)
                numbers = find_numbers(boxed_content)
                if numbers:
                    unthink_answer_strict = maybe_remove_comma(numbers[0])
                else:
                    unthink_answer_strict = maybe_remove_comma(boxed_content)
        
        # Extract answers using the relaxed method (same as GSM8K_generate_from_test.py)
        normal_answer_relaxed = maybe_remove_comma(find_number(re.sub(r'<think>.*?</think>', '', normal_response, flags=re.DOTALL)))
        unthink_answer_relaxed = maybe_remove_comma(find_number(re.sub(r'<think>.*?</think>', '', unthink_response, flags=re.DOTALL)))
        
        # Extract ground truth using the same relaxed method
        ground_truth_answer = maybe_remove_comma(find_number(ground_truth))
        
        # Track cases where boxed is wrong but relaxed method is right
        if normal_answer_strict and normal_answer_strict != ground_truth_answer and normal_answer_relaxed == ground_truth_answer:
            normal_boxed_wrong_but_find_number_right.append(resp_id)
            
        if unthink_answer_strict and unthink_answer_strict != ground_truth_answer and unthink_answer_relaxed == ground_truth_answer:
            unthink_boxed_wrong_but_find_number_right.append(resp_id)
        
        # Use relaxed answers for all comparisons to match GSM8K_generate_from_test.py
        normal_correct = normal_answer_relaxed == ground_truth_answer
        unthink_correct = unthink_answer_relaxed == ground_truth_answer

        if normal_correct and unthink_correct:
            both_right_count += 1
        elif normal_undefined and unthink_undefined:
            both_undefined.append(resp_id)
        elif not normal_correct and not unthink_correct:
            both_wrong_count += 1
        elif normal_undefined and unthink_correct:
            normal_undefined_unthink_right.append(resp_id)
        elif unthink_undefined and normal_correct:
            unthink_undefined_normal_right.append(resp_id)
        elif not normal_correct and unthink_correct:
            normal_wrong_unthink_right.append(resp_id)
        elif normal_correct and not unthink_correct:
            unthink_wrong_normal_right.append(resp_id)

    return (normal_wrong_unthink_right, unthink_wrong_normal_right, 
            normal_undefined_unthink_right, unthink_undefined_normal_right, 
            both_undefined, both_wrong_count, both_right_count, total_questions, filtered_ids,
            normal_boxed_wrong_but_find_number_right, unthink_boxed_wrong_but_find_number_right)

def save_results_to_files(results, output_dir):
    """Save results to text files and numpy arrays."""
    os.makedirs(output_dir, exist_ok=True)
    
    categories = [
        "normal_wrong_unthink_right",
        "unthink_wrong_normal_right",
        "normal_undefined_unthink_right",
        "unthink_undefined_normal_right",
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
    parser = argparse.ArgumentParser(description="Compare answers for GSM8K dataset.")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test",
                        help="Dataset split to use: train or test (used only if normal_dir and unthink_dir not specified)")
    parser.add_argument("--normal_dir", type=str, default=None,
                        help="Directory containing normal responses (overrides split-based default)")
    parser.add_argument("--unthink_dir", type=str, default=None,
                        help="Directory containing unthink responses (overrides split-based default)")
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
    if args.normal_dir and args.unthink_dir:
        normal_dir = args.normal_dir
        unthink_dir = args.unthink_dir
        
        # Determine output directory if not specified
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Determine parent directory of normal_dir and create meta_analysis subdirectory
            parent_dir = os.path.dirname(os.path.abspath(normal_dir))
            base_name = os.path.basename(os.path.normpath(unthink_dir))
            output_dir = os.path.join(parent_dir, "meta_analysis", base_name)
    else:
        # Use default directories based on split
        if split == "train":
            normal_dir = 'outputs/gsm8k_train/decoded_text'
            unthink_dir = 'outputs/unthink_gsm8k_train/decoded_text'
            output_dir = 'outputs/meta_analysis/gsm8k_train'
        else:
            normal_dir = 'outputs/gsm8k_test/decoded_text'
            unthink_dir = 'outputs/unthink_gsm8k_test/decoded_text'
            output_dir = 'outputs/meta_analysis/gsm8k_test'

    print(f"Comparing responses from:")
    print(f"  Normal directory: {normal_dir}")
    print(f"  Unthink directory: {unthink_dir}")
    print(f"Saving results to: {output_dir}")
    if args.max_questions:
        print(f"Processing up to {args.max_questions} questions")
        
    # Create mapping between response IDs and GSM8K dataset indices
    print("Creating mapping between response files and GSM8K dataset...")
    id_mapping, not_found = create_question_mapping(gsm8k_split, normal_dir)
    print(f"Found mappings for {len(id_mapping)} questions")
    print(f"Could not find mappings for {len(not_found)} questions")

    # Compare responses (only over questions where the unthink response has <= 5 words in its <think> block)
    results = compare_responses(normal_dir, unthink_dir, gsm8k_split, id_mapping, args.max_questions)

    # Output results
    def print_results(description, count, total):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{description}: {count} out of {total} ({percentage:.2f}%)")

    # Use results[7] to access total_questions
    total_questions = results[7]

    print("\n--- Standard Comparison Results ---")
    print_results("Questions normal got wrong but unthink got right", len(results[0]), total_questions)
    print_results("Questions unthink got wrong but normal got right", len(results[1]), total_questions)
    print_results("Questions normal was undefined but unthink got right", len(results[2]), total_questions)
    print_results("Questions unthink was undefined but normal got right", len(results[3]), total_questions)
    print_results("Questions both were undefined", len(results[4]), total_questions)
    print_results("Number of questions both got wrong", results[5], total_questions)
    print_results("Number of questions both got right", results[6], total_questions)
    
    # Print additional information about answer matching methods
    print("\n--- Answer Extraction Method Comparison ---")
    print_results("Normal: boxed wrong but find_number correct", len(results[9]), total_questions)
    print_results("Unthink: boxed wrong but find_number correct", len(results[10]), total_questions)
    
    # Print information about filtering
    print(f"\nExcluded {len(results[8])} questions where unthink response had >5 words in <think> block")

    # Calculate the number of questions each got right
    normal_right = results[6] + len(results[1]) + len(results[3])
    unthink_right = results[6] + len(results[0]) + len(results[2])

    # Print overall accuracy matching GSM8K_generate_from_test.py method
    print("\n--- Overall Accuracy (Using GSM8K_generate_from_test.py extraction method) ---")
    print_results("Normal accuracy", normal_right, total_questions)
    print_results("Unthink accuracy", unthink_right, total_questions)

    # Calculate the number of questions each got right excluding undefined
    # Include cases where the other model was undefined but this one was right
    normal_right_excluding_undefined = results[6] + len(results[1]) + len(results[3])
    unthink_right_excluding_undefined = results[6] + len(results[0]) + len(results[2])

    # Calculate the total number of non-undefined questions for each set
    total_non_undefined_normal = total_questions - len(results[4]) - len(results[2])
    total_non_undefined_unthink = total_questions - len(results[4]) - len(results[3])

    # Print the adjusted results with correct percentages
    print("\n--- Accuracy Excluding Undefined Questions ---")
    print_results("Number of questions normal got right excluding undefined", normal_right_excluding_undefined, total_non_undefined_normal)
    print_results("Number of questions unthink got right excluding undefined", unthink_right_excluding_undefined, total_non_undefined_unthink)

    # Save results to files
    save_results_to_files(results[:5], output_dir)  # Only save the lists, not the counts
    
    # Save the filtered question IDs (those excluded based on unthink chain-of-thought length) to a separate text file.
    filtered_ids = results[8]
    with open(os.path.join(output_dir, "filtered_questions.txt"), 'w') as f:
        for qid in filtered_ids:
            f.write(f"{qid}\n")