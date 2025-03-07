import os
import re
import glob
import argparse
import datetime

def extract_classification(text):
    """
    Extract the very latest YES or NO classification from the model output.
    Returns the last occurrence of YES or NO in the file.
    """
    # Initialize result to unknown
    latest_result = "UNKNOWN"
    latest_position = -1
    
    # Pattern to match standalone YES or NO 
    pattern = r'\b(YES|NO)\b'
    
    # Find all occurrences of YES or NO
    for match in re.finditer(pattern, text, re.IGNORECASE):
        # If this match is later in the file than our current result, update it
        if match.start() > latest_position:
            latest_position = match.start()
            latest_result = match.group(1).upper()
    
    return latest_result

def has_complete_cot(text):
    """
    Check if the file has a complete chain-of-thought structure.
    A complete chain-of-thought has both <think> and </think> tags.
    """
    has_opening = '<think>' in text
    has_closing = '</think>' in text
    return has_opening and has_closing

def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(data_dir=None, summary_dir=None):
    # Use provided data directory or default
    output_dir = data_dir if data_dir else 'outputs/penguin/decoded_text_reindexed'
    
    # Generate a timestamp for the summary folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up summary directory
    if not summary_dir:
        summary_dir = f'summary_results_{timestamp}'
    
    # Ensure summary directory exists
    summary_path = os.path.join(os.path.dirname(output_dir), summary_dir)
    ensure_dir(summary_path)
    
    # Create log files
    file_results_path = os.path.join(summary_path, 'file_results.txt')
    summary_path_all = os.path.join(summary_path, 'overall_summary.txt')
    excluded_files_path = os.path.join(summary_path, 'excluded_files.txt')
    
    # Open file to write file-by-file results
    file_results = open(file_results_path, 'w')
    file_results.write(f"FILE-BY-FILE RESULTS\n")
    file_results.write(f"===================\n\n")
    file_results.write(f"Data directory: {output_dir}\n")
    file_results.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Find all text files in the output directory
    text_files = glob.glob(os.path.join(output_dir, '*.txt'))
    
    if not text_files:
        file_results.write(f"No text files found in directory: {output_dir}\n")
        file_results.close()
        print(f"No text files found in directory: {output_dir}")
        return
    
    # The correct answer for "Can penguins fly?" is NO
    correct_answer = "NO"
    
    # Counters for results
    total_files = 0
    processed_files = 0
    incomplete_cot_files = 0
    correct_count = 0
    yes_count = 0
    no_count = 0
    unknown_count = 0
    
    # Dictionary to store results by temperature
    results_by_temp = {}
    
    # List to track files with incomplete chain-of-thoughts
    incomplete_cot_list = []
    
    # Print processing status
    print(f"Processing files from {output_dir}...")
    
    # Process each file
    for file_path in text_files:
        total_files += 1
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if the file has a complete chain-of-thought
            if not has_complete_cot(content):
                incomplete_cot_files += 1
                incomplete_cot_list.append(filename)
                file_results.write(f"Excluded: {filename} - Incomplete chain-of-thought\n")
                continue
            
            processed_files += 1
            
            # Extract temperature from filename (e.g., text_1_temp0_6.txt)
            temp_match = re.search(r'temp(\d+_\d+)', filename)
            temp = temp_match.group(1).replace('_', '.') if temp_match else "unknown"
            
            if temp not in results_by_temp:
                results_by_temp[temp] = {"total": 0, "correct": 0, "yes": 0, "no": 0, "unknown": 0}
            
            results_by_temp[temp]["total"] += 1
            
            # Extract the very latest YES/NO classification
            classification = extract_classification(content)
            
            # Update counters
            if classification == "YES":
                yes_count += 1
                results_by_temp[temp]["yes"] += 1
            elif classification == "NO":
                no_count += 1
                correct_count += 1  # NO is the correct answer
                results_by_temp[temp]["no"] += 1
                results_by_temp[temp]["correct"] += 1
            else:
                unknown_count += 1
                results_by_temp[temp]["unknown"] += 1
                
            file_results.write(f"File: {filename}, Classification: {classification}, Temperature: {temp}\n")
                
        except Exception as e:
            file_results.write(f"Error processing {filename}: {e}\n")
    
    # Close file results file
    file_results.close()
    
    # Write overall summary
    with open(summary_path_all, 'w') as summary_file:
        summary_file.write("OVERALL SUMMARY\n")
        summary_file.write("===============\n\n")
        summary_file.write(f"Data directory: {output_dir}\n")
        summary_file.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        summary_file.write(f"Total files found: {total_files}\n")
        summary_file.write(f"Files processed: {processed_files}\n")
        summary_file.write(f"Files excluded (incomplete chain-of-thought): {incomplete_cot_files}\n\n")
        
        if processed_files > 0:
            summary_file.write(f"Results for processed files:\n")
            summary_file.write(f"Correct answers (NO): {correct_count} ({correct_count/processed_files*100:.2f}% of processed)\n")
            summary_file.write(f"Incorrect answers (YES): {yes_count} ({yes_count/processed_files*100:.2f}% of processed)\n")
            summary_file.write(f"Unknown classifications: {unknown_count} ({unknown_count/processed_files*100:.2f}% of processed)\n\n")
        else:
            summary_file.write("No files were processed with complete chain-of-thoughts.\n\n")
        
        # Write results by temperature
        if results_by_temp:
            summary_file.write("Results by Temperature:\n")
            summary_file.write("=====================\n\n")
            for temp, counts in results_by_temp.items():
                if counts["total"] > 0:
                    correct_pct = counts["correct"] / counts["total"] * 100
                    yes_pct = counts["yes"] / counts["total"] * 100
                    unknown_pct = counts["unknown"] / counts["total"] * 100
                    
                    summary_file.write(f"Temperature {temp}:\n")
                    summary_file.write(f"  Total files: {counts['total']}\n")
                    summary_file.write(f"  Correct answers (NO): {counts['correct']} ({correct_pct:.2f}%)\n")
                    summary_file.write(f"  Incorrect answers (YES): {counts['yes']} ({yes_pct:.2f}%)\n")
                    summary_file.write(f"  Unknown: {counts['unknown']} ({unknown_pct:.2f}%)\n\n")
    
    # Write list of excluded files
    with open(excluded_files_path, 'w') as excluded_file:
        excluded_file.write("EXCLUDED FILES\n")
        excluded_file.write("==============\n\n")
        excluded_file.write(f"Files excluded due to incomplete chain-of-thought: {incomplete_cot_files}\n\n")
        
        if incomplete_cot_list:
            excluded_file.write("Files with incomplete chain-of-thought:\n")
            for i, filename in enumerate(incomplete_cot_list, 1):
                excluded_file.write(f"{i}. {filename}\n")
    
    # Print quick summary to console
    print("\n" + "="*50)
    print("QUICK SUMMARY")
    print("="*50)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with complete chain-of-thought: {processed_files} ({processed_files/total_files*100:.1f}%)")
    print(f"Files with incomplete chain-of-thought: {incomplete_cot_files} ({incomplete_cot_files/total_files*100:.1f}%)")
    
    if processed_files > 0:
        print("\nClassification Results:")
        print(f"  NO answers: {no_count} ({no_count/processed_files*100:.1f}%)")
        print(f"  YES answers: {yes_count} ({yes_count/processed_files*100:.1f}%)")
        print(f"  UNKNOWN: {unknown_count} ({unknown_count/processed_files*100:.1f}%)")
        
        # Print brief temperature summary
        if len(results_by_temp) > 0:
            print("\nResults by Temperature:")
            for temp, counts in sorted(results_by_temp.items()):
                if counts["total"] > 0:
                    correct_pct = counts["correct"] / counts["total"] * 100
                    print(f"  Temp {temp}: {counts['correct']}/{counts['total']} correct ({correct_pct:.1f}%)")
    
    print("\nDetailed results saved to:")
    print(f"- File-by-file results: {file_results_path}")
    print(f"- Overall summary: {summary_path_all}")
    print(f"- Excluded files list: {excluded_files_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse classification results from text files.')
    parser.add_argument('--data_dir', type=str, help='Directory containing the decoded text files')
    parser.add_argument('--summary_dir', type=str, help='Directory to save summary results (default: summary_results_TIMESTAMP)')
    args = parser.parse_args()
    
    main(args.data_dir, args.summary_dir) 