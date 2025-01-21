#!/bin/bash

pip install -r requirements.txt

SRC_LANG="lv"
TGT_LANG="en"
POSTFIX="" # _boosted-500k

subword-nmt apply-bpe -c /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG/codes.bpe < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/test.$SRC_LANG > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/test.bpe.$SRC_LANG
subword-nmt apply-bpe -c /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG/codes.bpe < /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/test.en > /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/test.bpe.en

fairseq-preprocess \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --srcdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-$SRC_LANG/dict.en.txt \
    --tgtdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-$SRC_LANG/dict.$SRC_LANG.txt \
    --testpref /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/test.bpe \
    --destdir /netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/final \
    --workers 20


OUTPUT_DIR="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$SRC_LANG-tatoeba/final"

fairseq-generate $OUTPUT_DIR \
    --fp16 \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --path checkpoints/transformer_${SRC_LANG}_${TGT_LANG}${POSTFIX}/checkpoint_best.pt \
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
