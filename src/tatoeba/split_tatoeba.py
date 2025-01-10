def split_tatoeba_file(input_file, output_en, output_mt):
    """
    Splits a Tatoeba dataset file into separate English and Maltese files.

    Args:
        input_file (str): Path to the input file containing English and Maltese sentences.
        output_en (str): Path to save the English sentences.
        output_mt (str): Path to save the Maltese sentences.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    english_sentences = []
    maltese_sentences = []

    for line in lines:
        parts = line.strip().split('\t')  # Split line by tab character
        if len(parts) >= 3:  # Ensure there are at least three parts (eng, mlt, sentence pair)
            english_sentences.append(parts[2])  # English sentence
            maltese_sentences.append(parts[3])  # Maltese sentence

    # Save English sentences
    with open(output_en, 'w', encoding='utf-8') as en_file:
        en_file.write('\n'.join(english_sentences) + '\n')

    # Save Maltese sentences
    with open(output_mt, 'w', encoding='utf-8') as mt_file:
        mt_file.write('\n'.join(maltese_sentences) + '\n')


if __name__ == "__main__":
    input_file = "/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/tatoeba-test-v2021-08-07.eng-mlt.txt"
    output_en = "tatoeba.en"
    output_mt = "tatoeba.mt"

    split_tatoeba_file(input_file, output_en, output_mt)

    print(f"English sentences saved to {output_en}")
    print(f"Maltese sentences saved to {output_mt}")
