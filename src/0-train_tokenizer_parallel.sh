#!/bin/bash

pip install -r requirements.txt

# Concatenate training data for both languages
cat /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-train.en /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-train.mt > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/train.both

# Learn BPE with 32000 merge operations
subword-nmt learn-bpe -s 32000 < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/train.both > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe

# Apply BPE to all files
for lang in en mt; do
    for split in train dev test; do
        subword-nmt apply-bpe -c /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-${split}.${lang} > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-${split}.bpe.${lang}
    done
done