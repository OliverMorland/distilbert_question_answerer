# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:04 2025

@author: OMorland
"""

import json
import re

# Define the input and output file names
input_file = "text_docs/extra_question_answerer_dataset.txt"
output_file = "question_answerer/datasets/extra_examples.json"

# Function to convert the input text data to the required JSON format using regex
def create_qa_dataset(input_file, output_file):
    dataset = []

    # Open the input file and read its contents
    with open(input_file, "r") as file:
        text = file.read()

    # Define the regex pattern to match the context and the answer
    pattern = r"CONTEXT:\s*(.*?)\s*ANSWER:\s*(.*?)\s*(?=CONTEXT:|$)"
    
    # Find all matches of the pattern
    matches = re.findall(pattern, text, re.DOTALL)

    # Process the matches
    for context, answer in matches:
        # Create the JSON entry for each context-answer pair
        dataset.append({
            "context": context.strip(),
            "question": "what is the courthouse address?",  # This can be customized
            "answers": {
                "text": [answer.strip()],
                "answer_start": [context.find(answer.strip())]
            }
        })

    # Write the dataset to a JSON file
    with open(output_file, "w") as outfile:
        json.dump(dataset, outfile, indent=4)

    print(f"Dataset successfully written to {output_file}")

# Run the function
create_qa_dataset(input_file, output_file)
