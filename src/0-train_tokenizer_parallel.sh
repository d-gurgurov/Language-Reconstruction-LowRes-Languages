#!/bin/bash

pip install -r requirements.txt

LANGUAGE="lv"

# Concatenate training data for both languages
cat /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/opus.en-$LANGUAGE-train.en /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/opus.en-$LANGUAGE-train.$LANGUAGE > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/train.both

# Learn BPE with 32000 merge operations
subword-nmt learn-bpe -s 32000 < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/train.both > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/codes.bpe

# Apply BPE to all files
for lang in en $LANGUAGE; do
    for split in train dev test; do
        subword-nmt apply-bpe -c /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/codes.bpe < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/opus.en-$LANGUAGE-${split}.${lang} > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/opus.en-$LANGUAGE-${split}.bpe.${lang}
    done
done