#!/bin/bash

pip install -r requirements.txt

SRC_LANG="en"
TGT_LANG="mt"

OUTPUT_DIR="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/${SRC_LANG}-${TGT_LANG}"

fairseq-generate $OUTPUT_DIR \
    --fp16 \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --path checkpoints/transformer_${SRC_LANG}_${TGT_LANG}/checkpoint_best.pt \
    --beam 4 \
    --remove-bpe \
    --max-len-a 1.2 \
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
