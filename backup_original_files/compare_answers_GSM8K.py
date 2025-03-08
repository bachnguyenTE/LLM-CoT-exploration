import os
import re
import numpy as np
import json
from datasets import load_dataset
import argparse

def find_numbers(x: str) -> list[str]:
    return re.compile(r'-?[\d,]*\.?\d+').findall(x)

def find_number(x: str, answer_delimiter: str = 'The answer is') -> str:
    if answer_delimiter in x:
        after = x.split(answer_delimiter, 1)[-1]
        nums = find_numbers(after)
        if nums:
            return nums[0]
    nums = find_numbers(x)
    if nums:
        return nums[-1]
    return ''

def maybe_remove_comma(x: str) -> str:
    return x.replace(',', '')

def extract_question(text: str) -> str:
    match = re.search(r'Q:\s*(.*?)(?=<|$)', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'Q:\s*(.*?)(?=\n\n|$)', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def load_responses(directory: str) -> dict[int, str]:
    responses = {}
    if not os.path.isdir(directory):
        return responses
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

def count_think_words(response: str) -> int:
    """
    Count how many whitespace-delimited words appear between <think> and </think>.
    If no properly formed block, returns 0.
    """
    start = response.find('<think>')
    end = response.rfind('</think>')
    if start == -1 or end == -1 or end < start:
        return 0
    content = response[start+len('<think>'): end]
    return len(content.split())

def create_question_mapping(gsm8k_data, response_dir: str) -> tuple[dict[int,int], list[int]]:
    responses = load_responses(response_dir)
    
    question_to_idx = {}
    for i, item in enumerate(gsm8k_data):
        shortq = item['question'][:100].lower().strip()
        question_to_idx[shortq] = i
    
    id_mapping = {}
    not_found = []
    
    for resp_id, resp_text in responses.items():
        q = extract_question(resp_text)
        if not q:
            continue
        shortq = q[:100].lower().strip()
        
        found = False
        for gsm_short, gsm_idx in question_to_idx.items():
            if shortq in gsm_short or gsm_short in shortq:
                id_mapping[resp_id] = gsm_idx
                found = True
                break
        if not found:
            not_found.append(resp_id)
    
    return id_mapping, not_found

def save_mapping_to_file(id_mapping: dict[int,int], filename: str) -> None:
    as_strings = {str(k): v for k, v in id_mapping.items()}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(as_strings, f)

def load_mapping_from_file(filename: str) -> dict[int,int]:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def compare_responses(
    normal_dir: str,
    unthink_dir: str,
    gsm8k_data,
    id_mapping: dict[int,int],
    max_questions: int = None
):
    """
    Compare normal vs unthink. 
    We skip unthink responses whose <think> block has >5 words.
    Returns:
      ( normal_wrong_unthink_right,
        unthink_wrong_normal_right,
        normal_undefined_unthink_right,
        unthink_undefined_normal_right,
        both_undefined,
        both_wrong_count,
        both_right_count,
        total_questions,
        normal_boxed_wrong_findnum_right,
        unthink_boxed_wrong_findnum_right,
        normal_right_total,
        unthink_right_total,
        normal_undefined_ids,
        unthink_undefined_ids,
        normal_correct_ids,
        unthink_correct_ids,
        skipped_count
      )
    """
    normal_resps = load_responses(normal_dir)
    unthink_resps = load_responses(unthink_dir)
    
    normal_wrong_unthink_right = []
    unthink_wrong_normal_right = []
    normal_undefined_unthink_right = []
    unthink_undefined_normal_right = []
    both_undefined = []
    
    both_wrong_count = 0
    both_right_count = 0
    total_questions = 0
    skipped_count = 0
    
    normal_boxed_wrong_findnum_right = []
    unthink_boxed_wrong_findnum_right = []
    
    normal_right_total = 0
    unthink_right_total = 0
    
    normal_undefined_ids = []
    unthink_undefined_ids = []
    normal_correct_ids = []
    unthink_correct_ids = []
    
    qids = list(id_mapping.keys())
    qids.sort()
    if max_questions and max_questions < len(qids):
        qids = qids[:max_questions]
    
    for resp_id in qids:
        idx = id_mapping[resp_id]
        ground_truth_text = gsm8k_data[idx]['answer']
        
        normal_text = normal_resps.get(resp_id, '')
        unthink_text = unthink_resps.get(resp_id, '')
        if not normal_text or not unthink_text:
            continue
        
        # Skip if unthink has >5 words in <think>
        if count_think_words(unthink_text) > 5:
            skipped_count += 1
            continue
        
        total_questions += 1
        
        normal_outside = re.sub(r'<think>.*?</think>', '', normal_text, flags=re.DOTALL)
        unthink_outside = re.sub(r'<think>.*?</think>', '', unthink_text, flags=re.DOTALL)
        
        normal_undefined = ('\\boxed{' not in normal_outside)
        unthink_undefined = ('\\boxed{' not in unthink_outside)
        
        if normal_undefined:
            normal_undefined_ids.append(resp_id)
        if unthink_undefined:
            unthink_undefined_ids.append(resp_id)
        
        normal_ans_relaxed = maybe_remove_comma(find_number(normal_outside))
        unthink_ans_relaxed = maybe_remove_comma(find_number(unthink_outside))
        ground_truth_answer = maybe_remove_comma(find_number(ground_truth_text))
        
        # Strict
        normal_ans_strict = ''
        if not normal_undefined:
            match_box = re.search(r'\\boxed\{([^}]+)\}', normal_outside)
            if match_box:
                raw_box = match_box.group(1)
                nums = find_numbers(raw_box)
                if nums:
                    normal_ans_strict = maybe_remove_comma(nums[0])
                else:
                    normal_ans_strict = maybe_remove_comma(raw_box)
        
        unthink_ans_strict = ''
        if not unthink_undefined:
            match_box = re.search(r'\\boxed\{([^}]+)\}', unthink_outside)
            if match_box:
                raw_box = match_box.group(1)
                nums = find_numbers(raw_box)
                if nums:
                    unthink_ans_strict = maybe_remove_comma(nums[0])
                else:
                    unthink_ans_strict = maybe_remove_comma(raw_box)
        
        normal_correct = (normal_ans_relaxed == ground_truth_answer)
        unthink_correct = (unthink_ans_relaxed == ground_truth_answer)
        
        if normal_correct:
            normal_right_total += 1
            normal_correct_ids.append(resp_id)
        if unthink_correct:
            unthink_right_total += 1
            unthink_correct_ids.append(resp_id)
        
        if normal_ans_strict and (normal_ans_strict != ground_truth_answer) and normal_correct:
            normal_boxed_wrong_findnum_right.append(resp_id)
        if unthink_ans_strict and (unthink_ans_strict != ground_truth_answer) and unthink_correct:
            unthink_boxed_wrong_findnum_right.append(resp_id)
        
        # Category
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
    
    return (
        normal_wrong_unthink_right,    # 0
        unthink_wrong_normal_right,    # 1
        normal_undefined_unthink_right,# 2
        unthink_undefined_normal_right,# 3
        both_undefined,                # 4
        both_wrong_count,              # 5
        both_right_count,              # 6
        total_questions,               # 7
        normal_boxed_wrong_findnum_right,  # 8
        unthink_boxed_wrong_findnum_right, # 9
        normal_right_total,            # 10
        unthink_right_total,           # 11
        normal_undefined_ids,          # 12
        unthink_undefined_ids,         # 13
        normal_correct_ids,            # 14
        unthink_correct_ids,           # 15
        skipped_count                  # 16
    )

def save_results_to_files(results, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    categories = [
        "normal_wrong_unthink_right",
        "unthink_wrong_normal_right",
        "normal_undefined_unthink_right",
        "unthink_undefined_normal_right",
        "both_undefined"
    ]
    for cat, data in zip(categories, results[:5]):
        fname = os.path.join(output_dir, f"{cat}.txt")
        with open(fname, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"{item}\n")
    
    arr_dict = {cat: np.array(data) for cat, data in zip(categories, results[:5])}
    np.save(os.path.join(output_dir, "results_dict.npy"), arr_dict)

def main():
    parser = argparse.ArgumentParser(description="Compare normal vs unthink on GSM8K (with 5-word CoT filter).")
    parser.add_argument("--split", choices=["train","test"], default="test")
    parser.add_argument("--normal_dir", type=str, default=None)
    parser.add_argument("--unthink_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--mapping_file", type=str, default=None)
    parser.add_argument("--save_mapping", action="store_true")
    args = parser.parse_args()
    
    gsm8k = load_dataset("gsm8k","main")
    if args.split == "train":
        gsm8k_split = gsm8k["train"]
    else:
        gsm8k_split = gsm8k["test"]
    
    if not args.normal_dir or not args.unthink_dir:
        if args.split == "train":
            normal_dir = "outputs/gsm8k_train/decoded_text"
            unthink_dir = "outputs/unthink_gsm8k_train/decoded_text"
            out_dir = "outputs/meta_analysis/gsm8k_train"
        else:
            normal_dir = "outputs/gsm8k_test/decoded_text"
            unthink_dir = "outputs/unthink_gsm8k_test/decoded_text"
            out_dir = "outputs/meta_analysis/gsm8k_test"
    else:
        normal_dir = args.normal_dir
        unthink_dir = args.unthink_dir
        out_dir = args.output_dir if args.output_dir else "outputs/meta_analysis"
    
    print(f"Comparing responses from:")
    print(f"  Normal directory: {normal_dir}")
    print(f"  Unthink directory: {unthink_dir}")
    print(f"Saving results to: {out_dir}")
    
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
    
    results = compare_responses(
        normal_dir,
        unthink_dir,
        gsm8k_split,
        mapping,
        max_questions=args.max_questions
    )
    
    (
        normal_wrong_unthink_right,
        unthink_wrong_normal_right,
        normal_undefined_unthink_right,
        unthink_undefined_normal_right,
        both_undefined,
        both_wrong_count,
        both_right_count,
        total_q,
        normal_boxed_wrong_findnum_right,
        unthink_boxed_wrong_findnum_right,
        normal_right_total,
        unthink_right_total,
        normal_undefined_ids,
        unthink_undefined_ids,
        normal_correct_ids,
        unthink_correct_ids,
        skipped_count
    ) = results
    
    def ratio(count, total_):
        if total_ == 0:
            return "0.00%"
        return f"{(count/total_)*100:.2f}%"
    
    print("\n--- Standard Comparison Results ---")
    print(f"Questions normal got wrong but unthink got right: {len(normal_wrong_unthink_right)} out of {total_q} ({ratio(len(normal_wrong_unthink_right), total_q)})")
    print(f"Questions unthink got wrong but normal got right: {len(unthink_wrong_normal_right)} out of {total_q} ({ratio(len(unthink_wrong_normal_right), total_q)})")
    print(f"Questions normal was undefined but unthink got right: {len(normal_undefined_unthink_right)} out of {total_q} ({ratio(len(normal_undefined_unthink_right), total_q)})")
    print(f"Questions unthink was undefined but normal got right: {len(unthink_undefined_normal_right)} out of {total_q} ({ratio(len(unthink_undefined_normal_right), total_q)})")
    print(f"Questions both were undefined: {len(both_undefined)} out of {total_q} ({ratio(len(both_undefined), total_q)})")
    print(f"Number of questions both got wrong: {both_wrong_count} out of {total_q} ({ratio(both_wrong_count, total_q)})")
    print(f"Number of questions both got right: {both_right_count} out of {total_q} ({ratio(both_right_count, total_q)})")
    
    print("\n--- Answer Extraction Method Comparison ---")
    print(f"Normal: boxed wrong but find_number correct: {len(normal_boxed_wrong_findnum_right)} out of {total_q} ({ratio(len(normal_boxed_wrong_findnum_right), total_q)})")
    print(f"Unthink: boxed wrong but find_number correct: {len(unthink_boxed_wrong_findnum_right)} out of {total_q} ({ratio(len(unthink_boxed_wrong_findnum_right), total_q)})")
    
    print("\n--- Overall Accuracy (Using GSM8K_generate_from_test.py extraction method) ---")
    print(f"Normal accuracy: {normal_right_total} out of {total_q} ({ratio(normal_right_total, total_q)})")
    print(f"Unthink accuracy: {unthink_right_total} out of {total_q} ({ratio(unthink_right_total, total_q)})")
    
    normal_defined_count = total_q - len(normal_undefined_ids)
    unthink_defined_count = total_q - len(unthink_undefined_ids)
    
    normal_correct_set = set(normal_correct_ids)
    normal_undefined_set = set(normal_undefined_ids)
    unthink_correct_set = set(unthink_correct_ids)
    unthink_undefined_set = set(unthink_undefined_ids)
    
    normal_defined_correct_set = normal_correct_set - normal_undefined_set
    unthink_defined_correct_set = unthink_correct_set - unthink_undefined_set
    
    normal_defined_correct = len(normal_defined_correct_set)
    unthink_defined_correct = len(unthink_defined_correct_set)
    
    if normal_defined_count > 0:
        normal_defined_acc = (normal_defined_correct/normal_defined_count)*100
    else:
        normal_defined_acc = 0.0
    
    if unthink_defined_count > 0:
        unthink_defined_acc = (unthink_defined_correct/unthink_defined_count)*100
    else:
        unthink_defined_acc = 0.0
    
    print("\n--- Accuracy Excluding Undefined Questions ---")
    print(f"Number of questions normal got right excluding undefined: {normal_defined_correct} out of {normal_defined_count} ({normal_defined_acc:.2f}%)")
    print(f"Number of questions unthink got right excluding undefined: {unthink_defined_correct} out of {unthink_defined_count} ({unthink_defined_acc:.2f}%)")
    
    print("\n--- Undefined Responses ---")
    print(f"Number of normal responses that are undefined: {len(normal_undefined_ids)} out of {total_q} ({ratio(len(normal_undefined_ids), total_q)})")
    print(f"Number of unthink responses that are undefined: {len(unthink_undefined_ids)} out of {total_q} ({ratio(len(unthink_undefined_ids), total_q)})")
    
    print(f"\nNumber of unthink responses skipped (CoT >5 words): {skipped_count}")
    
    save_results_to_files(results, out_dir)
    print("Done.")

if __name__ == "__main__":
    main()