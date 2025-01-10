#!/bin/bash

pip install -r /netscratch/dgurgurov/projects2024/mt_lrls/src/requirements.txt

python split_tatoeba.py

subword-nmt apply-bpe -c /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/tatoeba.mt > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/tatoeba.mt.bpe
subword-nmt apply-bpe -c /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/tatoeba.en > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/tatoeba.en.bpe

