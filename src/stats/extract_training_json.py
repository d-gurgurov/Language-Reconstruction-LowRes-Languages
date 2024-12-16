import json
import re

# Define the input and output file paths
input_file = "log_en_mt.txt"
output_file = "training_data.json"

# Initialize containers for extracted data
training_updates = []
validation_results = []
epoch_summaries = []

# Define regex patterns
update_pattern = re.compile(r'\{"epoch": \d+, "update": .+?\}')
validation_pattern = re.compile(r'\{"epoch": \d+, "valid_loss": .+?\}')
epoch_summary_pattern = re.compile(r'\{"epoch": \d+, "train_loss": .+?\}')

# Function to remove duplicate entries
def remove_duplicates(entries, key):
    seen = set()
    unique_entries = []
    for entry in entries:
        identifier = tuple(entry[k] for k in key)
        if identifier not in seen:
            seen.add(identifier)
            unique_entries.append(entry)
    return unique_entries

# Read and process the input file
with open(input_file, "r") as file:
    log_data = file.read()

    # Extract training updates
    training_updates = [json.loads(match.group()) for match in update_pattern.finditer(log_data)]

    # Extract validation results
    validation_results = [json.loads(match.group()) for match in validation_pattern.finditer(log_data)]

    # Extract epoch summaries
    epoch_summaries = [json.loads(match.group()) for match in epoch_summary_pattern.finditer(log_data)]

# Remove duplicate entries
training_updates = remove_duplicates(training_updates, key=["epoch", "update"])
validation_results = remove_duplicates(validation_results, key=["epoch", "valid_loss"])
epoch_summaries = remove_duplicates(epoch_summaries, key=["epoch", "train_loss"])

# Consolidate the extracted data into a dictionary
output_data = {
    "training_updates": training_updates,
    "validation_results": validation_results,
    "epoch_summaries": epoch_summaries,
}

# Save the extracted data to a JSON file
with open(output_file, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Extracted data has been saved to {output_file}")
