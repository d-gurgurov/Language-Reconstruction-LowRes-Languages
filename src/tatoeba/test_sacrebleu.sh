#!/bin/bash

# pip install -r requirements.txt

fairseq-preprocess \
    --source-lang mt --target-lang en \
    --srcdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.en.txt \
    --tgtdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.mt.txt \
    --testpref /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/test.bpe \
    --destdir /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/final \
    --workers 20

SRC_LANG="mt"
TGT_LANG="en"
POSTFIX="safecheck-200k"

OUTPUT_DIR="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-tatoeba/final"

fairseq-generate $OUTPUT_DIR \
    --fp16 \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --path checkpoints/transformer_${SRC_LANG}_${TGT_LANG}/checkpoint_best.pt \
    --beam 4 \
    --remove-bpe \
    --max-len-a 1.2 \
    --gen-subset test \
    --max-len-b 10 \
    --max-tokens 4096 \
    --batch-size 1 \
    --sacrebleu \
    --results-path $OUTPUT_DIR \
    --lenpen 0.6 

# > $OUTPUT_DIR/gen.out

grep ^H $OUTPUT_DIR/generate-test.txt | cut -f 3 | sacremoses detokenize > $OUTPUT_DIR/gen.detok.out
grep ^T $OUTPUT_DIR/generate-test.txt | cut -f 2 | sacremoses detokenize > $OUTPUT_DIR/gen.ref.detok
sacrebleu $OUTPUT_DIR/gen.ref.detok -i $OUTPUT_DIR/gen.detok.out -m bleu -b -w 4
