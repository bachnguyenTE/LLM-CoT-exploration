import os
import re
import glob

def extract_classification(text):
    """
    Extract the YES or NO classification from the model output.
    Tries multiple patterns to be robust against different formats.
    """
    # Pattern 1: Match YES or NO after </think> tag
    match = re.search(r'</think>\s*\n\s*(YES|NO)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: Look for YES or NO at the beginning of a line after thinking
    if '</think>' in text:
        after_think = text.split('</think>')[1]
        lines = after_think.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line == "YES" or line == "NO":
                return line
            # Match YES or NO if it's the first word in a line
            match = re.match(r'^(YES|NO)\b', line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    
    # Pattern 3: Look for YES or NO on any line
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line == "YES" or line == "NO":
            return line
        # Match standalone YES or NO surrounded by non-alphanumeric characters
        match = re.search(r'\b(YES|NO)\b', line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return "UNKNOWN"

def main():
    # Path to the directory containing decoded text files
    output_dir = 'outputs/penguin/decoded_text_reindexed'
    
    # Find all text files in the output directory
    text_files = glob.glob(os.path.join(output_dir, '*.txt'))
    
    # The correct answer for "Can penguins fly?" is NO
    correct_answer = "NO"
    
    # Counters for results
    total_files = 0
    correct_count = 0
    yes_count = 0
    no_count = 0
    unknown_count = 0
    
    # Dictionary to store results by temperature
    results_by_temp = {}
    
    # Process each file
    for file_path in text_files:
        total_files += 1
        filename = os.path.basename(file_path)
        
        # Extract temperature from filename (e.g., text_1_temp0_6.txt)
        temp_match = re.search(r'temp(\d+_\d+)', filename)
        temp = temp_match.group(1).replace('_', '.') if temp_match else "unknown"
        
        if temp not in results_by_temp:
            results_by_temp[temp] = {"total": 0, "correct": 0, "yes": 0, "no": 0, "unknown": 0}
        
        results_by_temp[temp]["total"] += 1
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract the YES/NO classification
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
                
            print(f"File: {filename}, Classification: {classification}, Temperature: {temp}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Print summary
    print("\nOverall Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Correct answers (NO): {correct_count} ({correct_count/total_files*100:.2f}% of total)")
    print(f"Incorrect answers (YES): {yes_count} ({yes_count/total_files*100:.2f}% of total)")
    print(f"Unknown classifications: {unknown_count} ({unknown_count/total_files*100:.2f}% of total)")
    
    # Print results by temperature
    print("\nResults by Temperature:")
    for temp, counts in results_by_temp.items():
        correct_pct = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        yes_pct = counts["yes"] / counts["total"] * 100 if counts["total"] > 0 else 0
        unknown_pct = counts["unknown"] / counts["total"] * 100 if counts["total"] > 0 else 0
        
        print(f"\nTemperature {temp}:")
        print(f"  Total files: {counts['total']}")
        print(f"  Correct answers (NO): {counts['correct']} ({correct_pct:.2f}%)")
        print(f"  Incorrect answers (YES): {counts['yes']} ({yes_pct:.2f}%)")
        print(f"  Unknown: {counts['unknown']} ({unknown_pct:.2f}%)")

if __name__ == "__main__":
    main() 