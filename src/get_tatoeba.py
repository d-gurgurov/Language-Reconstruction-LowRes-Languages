# Define file paths
english_file_path = '/netscratch/dgurgurov/thesis/mt_lrls/data/tatoeba-test-v2021-03-30.eng'
maltese_file_path = '/netscratch/dgurgurov/thesis/mt_lrls/data/tatoeba-test-v2021-03-30.mlt'
output_file_path = 'test_tatoeba_21_03.txt'

# Read the content of English sentences
with open(english_file_path, 'r', encoding='utf-8') as eng_file:
    english_sentences = eng_file.readlines()

# Read the content of Maltese sentences
with open(maltese_file_path, 'r', encoding='utf-8') as mlt_file:
    maltese_sentences = mlt_file.readlines()

# Ensure both files have the same number of sentences
if len(english_sentences) != len(maltese_sentences):
    raise ValueError("The number of sentences in the two files do not match.")

# Write the combined content to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Write the header
    output_file.write('eng\tmlt\teng_sent\tmlt_sent\n')
    
    # Write each sentence pair
    for idx, (eng_sent, mlt_sent) in enumerate(zip(english_sentences, maltese_sentences)):
        # Remove any extra whitespace/newline characters
        eng_sent = eng_sent.strip()
        mlt_sent = mlt_sent.strip()
        
        # Write the combined line
        output_file.write(f'{idx+1}\t{idx+1}\t{eng_sent}\t{mlt_sent}\n')

print(f"Combined file has been created at {output_file_path}")