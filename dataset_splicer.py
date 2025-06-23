import json
import os
import random

directories = os.listdir()
print(os.getcwd())
print("Directories: " + str(directories))

# Load dataset 1
with open('datasets/combined_dataset_2.json', 'r', encoding='utf-8') as f1:
    dataset1 = json.load(f1)

# Load dataset 2
with open('datasets/trafficking_dui_assault_examples.json', 'r', encoding='utf-8') as f2:
    dataset2 = json.load(f2)

# Combine the datasets
combined_dataset = dataset1 + dataset2

# Shuffle the combined dataset
random.shuffle(combined_dataset)

# Save the shuffled dataset
new_dataset_file = 'datasets/combined_dataset_3.json'
with open('datasets/combined_dataset_3.json', 'w', encoding='utf-8') as fout:
    json.dump(combined_dataset, fout, ensure_ascii=False, indent=4)

print(f"Combined dataset with {len(combined_dataset)} entries saved to '{new_dataset_file}'.")
