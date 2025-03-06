import os
import re
import numpy as np
from datasets import load_dataset
import argparse

def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    numbers = re.compile(r'-?[\d,]*\.?\d+').findall(x)
    return numbers

def find_number_outside_think(x: str) -> str:
    """Finds the most relevant number in a string outside of <think> tags."""
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

    # Default to last number in the string
    numbers = find_numbers(outside_think)
    if numbers:
        return numbers[-1]
    return ''

def maybe_remove_comma(x: str) -> str:
    return x.replace(',', '')

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

def compare_responses(normal_dir, unthink_dir, ground_truths):
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

    for task_id in range(0, 3600):  # Restricting to range 0-3599
        ground_truth = ground_truths.get(task_id)
        if ground_truth is None:
            continue

        normal_response = normal_responses.get(task_id, '')
        unthink_response = unthink_responses.get(task_id, '')
        # Exclude question if there are more than 5 words in the chain-of-thought block of the unthink response.
        if count_think_words(unthink_response) > 5:
            filtered_ids.append(task_id)
            continue

        total_questions += 1

        # Check if the answer is undefined (no \boxed{} found outside <think> tags)
        normal_undefined = '\\boxed{' not in re.sub(r'<think>.*?</think>', '', normal_response, flags=re.DOTALL)
        unthink_undefined = '\\boxed{' not in re.sub(r'<think>.*?</think>', '', unthink_response, flags=re.DOTALL)

        normal_answer = maybe_remove_comma(find_number_outside_think(normal_response))
        unthink_answer = maybe_remove_comma(find_number_outside_think(unthink_response))
        ground_truth_answer = maybe_remove_comma(find_number_outside_think(ground_truth))

        normal_correct = normal_answer == ground_truth_answer
        unthink_correct = unthink_answer == ground_truth_answer

        if normal_correct and unthink_correct:
            both_right_count += 1
        elif normal_undefined and unthink_undefined:
            both_undefined.append(task_id)
        elif not normal_correct and not unthink_correct:
            both_wrong_count += 1
        elif normal_undefined and unthink_correct:
            normal_undefined_unthink_right.append(task_id)
        elif unthink_undefined and normal_correct:
            unthink_undefined_normal_right.append(task_id)
        elif not normal_correct and unthink_correct:
            normal_wrong_unthink_right.append(task_id)
        elif normal_correct and not unthink_correct:
            unthink_wrong_normal_right.append(task_id)

    return (normal_wrong_unthink_right, unthink_wrong_normal_right, 
            normal_undefined_unthink_right, unthink_undefined_normal_right, 
            both_undefined, both_wrong_count, both_right_count, total_questions, filtered_ids)

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
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train",
                        help="Dataset split to use: train or test")
    args = parser.parse_args()
    split = args.split

    # Load ground truths and set directories based on the split
    gsm8k = load_dataset("gsm8k", "main")
    if split == "train":
        gsm8k_split = gsm8k['train']
        normal_dir = 'outputs/gsm8k_train/decoded_text'
        unthink_dir = 'outputs/unthink_gsm8k_train/decoded_text'
        output_dir = 'outputs/meta_analysis/gsm8k_train'
    else:
        gsm8k_split = gsm8k['test']
        normal_dir = 'outputs/gsm8k_test/decoded_text'
        unthink_dir = 'outputs/unthink_gsm8k_test/decoded_text'
        output_dir = 'outputs/meta_analysis/gsm8k_test'

    ground_truths = {i: problem['answer'] for i, problem in enumerate(gsm8k_split)}

    # Compare responses (only over questions where the unthink response has <= 5 words in its <think> block)
    results = compare_responses(normal_dir, unthink_dir, ground_truths)

    # Output results
    def print_results(description, count, total):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{description}: {count} out of {total} ({percentage:.2f}%)")

    print_results("Questions normal got wrong but unthink got right", len(results[0]), results[-2])
    print_results("Questions unthink got wrong but normal got right", len(results[1]), results[-2])
    print_results("Questions normal was undefined but unthink got right", len(results[2]), results[-2])
    print_results("Questions unthink was undefined but normal got right", len(results[3]), results[-2])
    print_results("Questions both were undefined", len(results[4]), results[-2])
    print_results("Number of questions both got wrong", results[5], results[-2])
    print_results("Number of questions both got right", results[6], results[-2])

    # Calculate the number of questions each got right excluding undefined
    normal_right_excluding_undefined = results[6] + len(results[1]) + len(results[3])
    unthink_right_excluding_undefined = results[6] + len(results[0]) + len(results[2])

    # Calculate the total number of non-undefined questions for each set
    total_non_undefined_normal = results[-2] - (len(results[4]) + len(results[2]))
    total_non_undefined_unthink = results[-2] - (len(results[4]) + len(results[3]))

    # Print the adjusted results with correct percentages
    print_results("Number of questions normal got right excluding undefined", normal_right_excluding_undefined, total_non_undefined_normal)
    print_results("Number of questions unthink got right excluding undefined", unthink_right_excluding_undefined, total_non_undefined_unthink)

    # Save results to files
    save_results_to_files(results[:5], output_dir)  # Only save the lists, not the counts

    # Save the filtered question IDs (those excluded based on unthink chain-of-thought length) to a separate text file.
    filtered_ids = results[8]
    with open(os.path.join(output_dir, "filtered_questions.txt"), 'w') as f:
        for qid in filtered_ids:
            f.write(f"{qid}\n")