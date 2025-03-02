import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
from parse_penguin_results import extract_classification

def visualize_results():
    # Path to the directory containing decoded text files
    output_dir = 'outputs/penguin/decoded_text_reindexed'
    
    # Find all text files in the output directory
    text_files = glob.glob(os.path.join(output_dir, '*.txt'))
    
    # Dictionary to store results by temperature
    results_by_temp = {}
    
    # Process each file
    for file_path in text_files:
        filename = os.path.basename(file_path)
        
        # Extract temperature from filename (e.g., text_1_temp0_6.txt)
        temp_match = re.search(r'temp(\d+_\d+)', filename)
        temp = temp_match.group(1).replace('_', '.') if temp_match else "unknown"
        
        if temp not in results_by_temp:
            results_by_temp[temp] = {"total": 0, "yes": 0, "no": 0, "unknown": 0}
        
        results_by_temp[temp]["total"] += 1
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract the YES/NO classification
            classification = extract_classification(content)
            
            # Update counters
            if classification == "YES":
                results_by_temp[temp]["yes"] += 1
            elif classification == "NO":
                results_by_temp[temp]["no"] += 1
            else:
                results_by_temp[temp]["unknown"] += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create visualizations
    create_bar_chart(results_by_temp)
    create_pie_chart(results_by_temp)
    
    print("Visualizations created successfully!")

def create_bar_chart(results_by_temp):
    """Create a bar chart showing results by temperature"""
    # Prepare data
    temps = []
    correct_percentages = []
    incorrect_percentages = []
    unknown_percentages = []
    
    for temp, counts in sorted(results_by_temp.items(), key=lambda x: float(x[0]) if x[0] != "unknown" else 999):
        temps.append(temp)
        total = counts["total"]
        
        if total > 0:
            correct_percentages.append(counts["no"] / total * 100)
            incorrect_percentages.append(counts["yes"] / total * 100)
            unknown_percentages.append(counts["unknown"] / total * 100)
        else:
            correct_percentages.append(0)
            incorrect_percentages.append(0)
            unknown_percentages.append(0)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar positions
    x = np.arange(len(temps))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, correct_percentages, width, label='Correct (NO)')
    plt.bar(x, incorrect_percentages, width, label='Incorrect (YES)')
    plt.bar(x + width, unknown_percentages, width, label='Unknown')
    
    # Add labels and title
    plt.ylabel('Percentage (%)')
    plt.xlabel('Temperature')
    plt.title('Classification Accuracy by Temperature')
    plt.xticks(x, temps)
    plt.ylim(0, 100)
    
    # Add a grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Add value labels on top of the bars
    for i, v in enumerate(correct_percentages):
        plt.text(i - width, v + 2, f"{v:.1f}%", ha='center')
    for i, v in enumerate(incorrect_percentages):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    for i, v in enumerate(unknown_percentages):
        plt.text(i + width, v + 2, f"{v:.1f}%", ha='center')
    
    # Save figure
    os.makedirs('outputs/visualizations', exist_ok=True)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/classification_by_temperature.png')
    plt.close()

def create_pie_chart(results_by_temp):
    """Create a pie chart showing overall results"""
    # Aggregate data across all temperatures
    total_correct = sum(counts["no"] for counts in results_by_temp.values())
    total_incorrect = sum(counts["yes"] for counts in results_by_temp.values())
    total_unknown = sum(counts["unknown"] for counts in results_by_temp.values())
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    labels = ['Correct (NO)', 'Incorrect (YES)', 'Unknown']
    sizes = [total_correct, total_incorrect, total_unknown]
    colors = ['#66b3ff', '#ff9999', '#99ff99']
    explode = (0.1, 0, 0)  # explode the 1st slice (Correct)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    
    # Add title
    plt.title('Overall Classification Results', fontsize=16)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('outputs/visualizations/overall_results_pie.png')
    plt.close()

if __name__ == "__main__":
    visualize_results() 