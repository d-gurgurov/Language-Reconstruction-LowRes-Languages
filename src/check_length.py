import csv
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Training a baseline on full data')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}


def calculate_average_text_length(csv_file, column_name):
    total_length = 0
    count = 0

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Use DictReader to read rows as dictionaries
        for row in reader:
            text = row[column_name]
            total_length += len(text)
            count += 1

    if count == 0:  # Avoid division by zero
        return 0

    average_length = total_length / count
    return average_length

# Usage example
csv_file_path = f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_goodlrl_1.csv'
column_name_1 = "translated_text"
column_name_2 = 'good_lrl' 

print("Language:", language)

average_length = calculate_average_text_length(csv_file_path, "original_text")
print(f"The average length of texts in the 'original_text' column is: {average_length:.2f}")

average_length = calculate_average_text_length(csv_file_path, column_name_1)
print(f"The average length of texts in the '{column_name_1}' column is: {average_length:.2f}")

average_length = calculate_average_text_length(csv_file_path, column_name_2)
print(f"The average length of texts in the '{column_name_2}' column is: {average_length:.2f}")