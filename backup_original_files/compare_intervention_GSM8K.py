import os
import re
import numpy as np
import json
from datasets import load_dataset
import argparse

def find_numbers(x: str) -> list[str]:
    """Find all numeric substrings (including negative, decimals, comma-separated)."""
    return re.compile(r'-?[\d,]*\.?\d+').findall(x)

def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
    """
    Implement the "relaxed" answer-extraction logic from GSM8K_generate_from_test.py:
      - If we see "The answer is", parse the *first* number after that phrase.
      - Otherwise, parse the *last* number found in the entire string.
    """
    if answer_delimiter in x:
        remainder = x.split(answer_delimiter, 1)[-1]
        nums = find_numbers(remainder)
        if nums:
            return nums[0]
    nums = find_numbers(x)
    if nums:
        return nums[-1]
    return ''

def maybe_remove_comma(x: str) -> str:
    """Remove commas from a numeric string, e.g. '1,000' -> '1000'."""
    return x.replace(',', '')

def extract_question(text: str) -> str:
    """
    Extract the question portion from 'Q: ...' in the response text.
    We'll compare it to the GSM8K dataset question to find a mapping.
    """
    # First attempt: capture from 'Q:' up to the next < or end-of-string
    match = re.search(r'Q:\s*(.*?)(?=<|$)', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: from 'Q:' up to blank line or end-of-string
    match = re.search(r'Q:\s*(.*?)(?=\n\n|$)', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def load_responses(directory: str) -> dict[int, str]:
    """
    Read all *.txt files in the directory, parse an integer ID from the
    filename (e.g. 'normal_1234.txt' -> ID=1234), and return {ID -> file contents}.
    """
    responses = {}
    if not os.path.isdir(directory):
        return responses  # no files if the directory doesn't exist
    for fname in os.listdir(directory):
        if fname.endswith('.txt'):
            try:
                tid_str = fname.split('_')[-1].split('.')[0]
                tid = int(tid_str)
            except ValueError:
                continue
            fullpath = os.path.join(directory, fname)
            with open(fullpath, 'r', encoding='utf-8') as f:
                responses[tid] = f.read()
    return responses

def create_question_mapping(gsm8k_data, response_dir: str) -> tuple[dict[int,int], list[int]]:
    """
    Attempt to map each local response ID -> an index in the GSM8K dataset.
    We do this by:
      1) Extracting the 'Q:' portion from the response.
      2) Looking for partial string matches against the first ~100 chars of each GSM8K question.
    Returns:
      id_mapping = {local_response_id: gsm8k_index}
      not_found  = list of IDs that could not be mapped
    """
    responses = load_responses(response_dir)
    
    # Pre-build a dictionary of the first 100 chars of each GSM8K question
    question_to_idx = {}
    for i, item in enumerate(gsm8k_data):
        q_text = item['question']
        shortq = q_text[:100].lower().strip()
        question_to_idx[shortq] = i
    
    id_mapping = {}
    not_found = []
    
    for resp_id, resp_text in responses.items():
        extracted_q = extract_question(resp_text)
        if not extracted_q:
            continue
        shortresp = extracted_q[:100].lower().strip()
        
        found_match = False
        # Attempt a substring match in either direction
        for gsm_short, gsm_idx in question_to_idx.items():
            if shortresp in gsm_short or gsm_short in shortresp:
                id_mapping[resp_id] = gsm_idx
                found_match = True
                break
        if not found_match:
            not_found.append(resp_id)
    
    return id_mapping, not_found

def save_mapping_to_file(id_mapping: dict[int,int], filename: str) -> None:
    """Save ID->index mapping to JSON (keys as strings)."""
    as_strings = {str(k): v for k, v in id_mapping.items()}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(as_strings, f)

def load_mapping_from_file(filename: str) -> dict[int,int]:
    """Load ID->index mapping from JSON (keys as strings)."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def compare_responses(
    normal_dir: str,
    intervention_dir: str,
    gsm8k_data,
    id_mapping: dict[int,int],
    max_questions: int = None
):
    """
    Compare 'normal' vs 'intervention' responses, with NO filter for chain-of-thought length.
    Detect undefined (no \boxed{} outside <think>), check correctness via "relaxed" extraction.
    
    Returns a tuple of:
      (
        normal_wrong_intervention_right,       # list of IDs
        intervention_wrong_normal_right,       # list of IDs
        normal_undefined_intervention_right,   # list of IDs
        intervention_undefined_normal_right,   # list of IDs
        both_undefined,                        # list of IDs
        both_wrong_count,                      # int
        both_right_count,                      # int
        total_questions,                       # int
        normal_boxed_wrong_but_find_number_right,   # list of IDs
        intervention_boxed_wrong_but_find_number_right, # list of IDs
        normal_right_total,                    # int (# correct by normal)
        intervention_right_total,              # int (# correct by intervention)
        normal_undefined_ids,                  # list of IDs
        intervention_undefined_ids,            # list of IDs
        normal_correct_ids,                    # list of IDs
        intervention_correct_ids               # list of IDs
      )
    """
    normal_resps = load_responses(normal_dir)
    inter_resps = load_responses(intervention_dir)
    
    normal_wrong_inter_right = []
    inter_wrong_normal_right = []
    normal_undef_inter_right = []
    inter_undef_normal_right = []
    both_undefined = []
    both_wrong_count = 0
    both_right_count = 0
    total_questions = 0
    
    # For strict vs relaxed analysis
    normal_boxed_wrong_but_find_number_right = []
    intervention_boxed_wrong_but_find_number_right = []
    
    # Tally how many times normal or intervention is correct (relaxed)
    normal_right_total = 0
    intervention_right_total = 0
    
    # Track exactly which IDs are undefined, and which are correct
    normal_undefined_ids = []
    intervention_undefined_ids = []
    normal_correct_ids = []
    intervention_correct_ids = []
    
    qids = list(id_mapping.keys())
    qids.sort()
    if max_questions and max_questions < len(qids):
        qids = qids[:max_questions]
    
    for resp_id in qids:
        idx = id_mapping[resp_id]
        ground_truth_text = gsm8k_data[idx]['answer']
        
        normal_text = normal_resps.get(resp_id, '')
        inter_text = inter_resps.get(resp_id, '')
        
        # If either is missing, skip
        if not normal_text or not inter_text:
            continue
        
        total_questions += 1
        
        # Check undefined (no \boxed outside <think>)
        normal_outside_think = re.sub(r'<think>.*?</think>', '', normal_text, flags=re.DOTALL)
        inter_outside_think = re.sub(r'<think>.*?</think>', '', inter_text, flags=re.DOTALL)
        
        normal_undefined = ('\\boxed{' not in normal_outside_think)
        intervention_undefined = ('\\boxed{' not in inter_outside_think)
        
        if normal_undefined:
            normal_undefined_ids.append(resp_id)
        if intervention_undefined:
            intervention_undefined_ids.append(resp_id)
        
        # Relaxed extraction
        normal_ans_relaxed = maybe_remove_comma(find_number(normal_outside_think))
        inter_ans_relaxed = maybe_remove_comma(find_number(inter_outside_think))
        ground_truth_answer = maybe_remove_comma(find_number(ground_truth_text))
        
        # Strict extraction (for analysis)
        normal_ans_strict = ''
        if not normal_undefined:
            match_boxed = re.search(r'\\boxed\{([^}]+)\}', normal_outside_think)
            if match_boxed:
                raw_box = match_boxed.group(1)
                nums = find_numbers(raw_box)
                if nums:
                    normal_ans_strict = maybe_remove_comma(nums[0])
                else:
                    normal_ans_strict = maybe_remove_comma(raw_box)
        
        intervention_ans_strict = ''
        if not intervention_undefined:
            match_boxed = re.search(r'\\boxed\{([^}]+)\}', inter_outside_think)
            if match_boxed:
                raw_box = match_boxed.group(1)
                nums = find_numbers(raw_box)
                if nums:
                    intervention_ans_strict = maybe_remove_comma(nums[0])
                else:
                    intervention_ans_strict = maybe_remove_comma(raw_box)
        
        # Check correctness (relaxed)
        normal_correct = (normal_ans_relaxed == ground_truth_answer)
        inter_correct = (inter_ans_relaxed == ground_truth_answer)
        
        if normal_correct:
            normal_right_total += 1
            normal_correct_ids.append(resp_id)
        if inter_correct:
            intervention_right_total += 1
            intervention_correct_ids.append(resp_id)
        
        # If strict is different from ground-truth but relaxed is correct
        if normal_ans_strict and (normal_ans_strict != ground_truth_answer) and normal_correct:
            normal_boxed_wrong_but_find_number_right.append(resp_id)
        if intervention_ans_strict and (intervention_ans_strict != ground_truth_answer) and inter_correct:
            intervention_boxed_wrong_but_find_number_right.append(resp_id)
        
        # Now classify into exactly one outcome bucket
        if normal_correct and inter_correct:
            both_right_count += 1
        elif normal_undefined and intervention_undefined:
            both_undefined.append(resp_id)
        elif not normal_correct and not inter_correct:
            both_wrong_count += 1
        elif normal_undefined and inter_correct:
            normal_undef_inter_right.append(resp_id)
        elif intervention_undefined and normal_correct:
            inter_undef_normal_right.append(resp_id)
        elif not normal_correct and inter_correct:
            normal_wrong_inter_right.append(resp_id)
        elif normal_correct and not inter_correct:
            inter_wrong_normal_right.append(resp_id)
    
    return (
        normal_wrong_inter_right,                  # 0
        inter_wrong_normal_right,                  # 1
        normal_undef_inter_right,                  # 2
        inter_undef_normal_right,                  # 3
        both_undefined,                            # 4
        both_wrong_count,                          # 5
        both_right_count,                          # 6
        total_questions,                           # 7
        normal_boxed_wrong_but_find_number_right,  # 8
        intervention_boxed_wrong_but_find_number_right, # 9
        normal_right_total,                        # 10
        intervention_right_total,                  # 11
        normal_undefined_ids,                      # 12
        intervention_undefined_ids,                # 13
        normal_correct_ids,                        # 14
        intervention_correct_ids                   # 15
    )

def save_results_to_files(results, output_dir: str) -> None:
    """
    Save the first 5 lists to .txt files, also as a .npy dictionary, just like before.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    categories = [
        "normal_wrong_intervention_right",
        "intervention_wrong_normal_right",
        "normal_undefined_intervention_right",
        "intervention_undefined_normal_right",
        "both_undefined"
    ]
    for cat, data in zip(categories, results[:5]):
        path_txt = os.path.join(output_dir, f"{cat}.txt")
        with open(path_txt, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"{item}\n")
    
    array_dict = {cat: np.array(data) for cat, data in zip(categories, results[:5])}
    np.save(os.path.join(output_dir, "results_dict.npy"), array_dict)

def main():
    parser = argparse.ArgumentParser(description="Compare normal vs intervention on GSM8K (no 5-word CoT filter).")
    parser.add_argument("--split", choices=["train","test"], default="test",
                        help="Whether to use train or test split of GSM8K if no dirs are specified.")
    parser.add_argument("--normal_dir", type=str, default=None,
                        help="Directory containing normal responses.")
    parser.add_argument("--intervention_dir", type=str, default=None,
                        help="Directory containing intervention responses.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results; if None, create a default.")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Optionally limit how many questions to evaluate.")
    parser.add_argument("--mapping_file", type=str, default=None,
                        help="Path to a JSON file for ID->GSM8K index mapping.")
    parser.add_argument("--save_mapping", action="store_true",
                        help="If set, save a newly created mapping to disk.")
    args = parser.parse_args()
    
    # 1) Load GSM8K
    gsm8k = load_dataset("gsm8k","main")
    if args.split == "train":
        gsm8k_split = gsm8k["train"]
    else:
        gsm8k_split = gsm8k["test"]
    
    # 2) Determine directories or defaults
    if not args.normal_dir or not args.intervention_dir:
        if args.split == "train":
            normal_dir = "outputs/gsm8k_train/decoded_text"
            intervention_dir = "outputs/intervention_gsm8k_train/decoded_text"
            out_dir = "outputs/meta_analysis/intervention_gsm8k_train"
        else:
            normal_dir = "outputs/gsm8k_test/decoded_text"
            intervention_dir = "outputs/intervention_gsm8k_test/decoded_text"
            out_dir = "outputs/meta_analysis/intervention_gsm8k_test"
    else:
        normal_dir = args.normal_dir
        intervention_dir = args.intervention_dir
        if args.output_dir is None:
            # Use a default location if not provided
            parent = os.path.dirname(os.path.abspath(normal_dir))
            base = os.path.basename(os.path.normpath(intervention_dir))
            out_dir = os.path.join(parent, "meta_analysis", base)
        else:
            out_dir = args.output_dir
    
    # 3) Print an initial summary matching your requested style
    print(f"Comparing responses from:")
    print(f"  Normal directory: {normal_dir}")
    print(f"  Intervention directory: {intervention_dir}")
    print(f"Saving results to: {out_dir}")
    
    # 4) Load or build ID->index mapping
    print("Creating mapping between response files and GSM8K dataset...")
    if args.mapping_file and os.path.exists(args.mapping_file):
        mapping = load_mapping_from_file(args.mapping_file)
        print(f"Found mappings for {len(mapping)} questions")
        print(f"Could not find mappings for 0 questions (pre-existing file).")
    else:
        mapping, not_found = create_question_mapping(gsm8k_split, normal_dir)
        print(f"Found mappings for {len(mapping)} questions")
        print(f"Could not find mappings for {len(not_found)} questions")
        if args.mapping_file and args.save_mapping:
            save_mapping_to_file(mapping, args.mapping_file)
    
    # 5) Compare
    results = compare_responses(
        normal_dir,
        intervention_dir,
        gsm8k_split,
        mapping,
        max_questions=args.max_questions
    )
    
    # Unpack results
    (
        normal_wrong_inter_right,
        inter_wrong_normal_right,
        normal_undef_inter_right,
        inter_undef_normal_right,
        both_undefined,
        both_wrong_count,
        both_right_count,
        total_q,
        normal_boxed_wrong_findnum_right,
        inter_boxed_wrong_findnum_right,
        normal_right_total,
        intervention_right_total,
        normal_undefined_ids,
        intervention_undefined_ids,
        normal_correct_ids,
        intervention_correct_ids
    ) = results
    
    # 6) Summaries
    # Standard comparison
    def ratio(count, total):
        if total == 0:
            return "0.00%"
        return f"{(count/total)*100:.2f}%"
    
    print("\n--- Standard Comparison Results ---")
    print(f"Questions normal got wrong but intervention got right: {len(normal_wrong_inter_right)} out of {total_q} ({ratio(len(normal_wrong_inter_right), total_q)})")
    print(f"Questions intervention got wrong but normal got right: {len(inter_wrong_normal_right)} out of {total_q} ({ratio(len(inter_wrong_normal_right), total_q)})")
    print(f"Questions normal was undefined but intervention got right: {len(normal_undef_inter_right)} out of {total_q} ({ratio(len(normal_undef_inter_right), total_q)})")
    print(f"Questions intervention was undefined but normal got right: {len(inter_undef_normal_right)} out of {total_q} ({ratio(len(inter_undef_normal_right), total_q)})")
    print(f"Questions both were undefined: {len(both_undefined)} out of {total_q} ({ratio(len(both_undefined), total_q)})")
    print(f"Number of questions both got wrong: {both_wrong_count} out of {total_q} ({ratio(both_wrong_count, total_q)})")
    print(f"Number of questions both got right: {both_right_count} out of {total_q} ({ratio(both_right_count, total_q)})")
    
    # Strict vs. relaxed
    print("\n--- Answer Extraction Method Comparison ---")
    print(f"Normal: boxed wrong but find_number correct: {len(normal_boxed_wrong_findnum_right)} out of {total_q} ({ratio(len(normal_boxed_wrong_findnum_right), total_q)})")
    print(f"Intervention: boxed wrong but find_number correct: {len(inter_boxed_wrong_findnum_right)} out of {total_q} ({ratio(len(inter_boxed_wrong_findnum_right), total_q)})")
    
    # Overall accuracy
    print("\n--- Overall Accuracy (Using GSM8K_generate_from_test.py extraction method) ---")
    print(f"Normal accuracy: {normal_right_total} out of {total_q} ({ratio(normal_right_total, total_q)})")
    print(f"Intervention accuracy: {intervention_right_total} out of {total_q} ({ratio(intervention_right_total, total_q)})")
    
    # Accuracy excluding undefined
    normal_defined_set = set(range(total_q)) - set(normal_undefined_ids)  # This doesn't quite work, we need the actual IDs
    # Actually let's gather the actual set of QIDs we used:
    all_qids = set(mapping.keys())
    # Filter out the ones we never actually compared (missing normal or intervention?)
    # but for simplicity, let's rely on the final total_q to mean we included them.
    
    # We'll just do:
    normal_undefined_set = set(normal_undefined_ids)
    intervention_undefined_set = set(intervention_undefined_ids)
    # The set of qids we actually compared is "all compared qids"
    # We'll build that in the loop if we wanted to be precise. Let's just assume it's the same as mapping so we do:
    # Actually let's track them from the sum of correct_ids + the other categories. Let's do it a simpler way:
    
    normal_defined_count = total_q - len(normal_undefined_ids)
    intervention_defined_count = total_q - len(intervention_undefined_ids)
    
    # Now how many normal got right among the "defined" subset?
    # We need to see which correct IDs are not undefined
    normal_correct_set = set(normal_correct_ids)
    normal_defined_correct_set = normal_correct_set - normal_undefined_set
    # same for intervention
    intervention_correct_set = set(intervention_correct_ids)
    intervention_defined_correct_set = intervention_correct_set - intervention_undefined_set
    
    normal_defined_correct = len(normal_defined_correct_set)
    intervention_defined_correct = len(intervention_defined_correct_set)
    
    print("\n--- Accuracy Excluding Undefined Questions ---")
    # If normal_defined_count = 0, we avoid division by zero
    if normal_defined_count > 0:
        normal_defined_acc = (normal_defined_correct / normal_defined_count)*100
    else:
        normal_defined_acc = 0.0
    
    if intervention_defined_count > 0:
        intervention_defined_acc = (intervention_defined_correct / intervention_defined_count)*100
    else:
        intervention_defined_acc = 0.0
    
    print(f"Number of questions normal got right excluding undefined: {normal_defined_correct} out of {normal_defined_count} ({normal_defined_acc:.2f}%)")
    print(f"Number of questions intervention got right excluding undefined: {intervention_defined_correct} out of {intervention_defined_count} ({intervention_defined_acc:.2f}%)")
    
    print("\n--- Undefined Responses ---")
    print(f"Number of normal responses that are undefined: {len(normal_undefined_ids)} out of {total_q} ({ratio(len(normal_undefined_ids), total_q)})")
    print(f"Number of intervention responses that are undefined: {len(intervention_undefined_ids)} out of {total_q} ({ratio(len(intervention_undefined_ids), total_q)})")
    
    # 7) Save the first 5 categories to files
    save_results_to_files(results, out_dir)
    print("Done.")

if __name__ == "__main__":
    main()