#!/bin/bash

pip install -r requirements.txt

# Set up directories and file paths
TYPE="500k"
OUT_DIR="data-bin/mt-en-boosted-$TYPE"
mkdir -p $OUT_DIR

BASE_CORPUS="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt"
NEWSCRAWL_EN="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/newscrawl/newscrawl.1m.en"
NEWSCRAWL_MT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/reconstruct/reordered.newscrawl.boost.1m.mt"
BPE_CODES="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe"

# Number of lines to take from NewsCrawl
NUM_LINES=500000

echo "Preparing training data by combining original corpus with NewsCrawl..."
cat $BASE_CORPUS-train.en <(head -n $NUM_LINES $NEWSCRAWL_EN) > $OUT_DIR/train.en
cat $BASE_CORPUS-train.mt <(head -n $NUM_LINES $NEWSCRAWL_MT) > $OUT_DIR/train.mt


echo "Copying validation and test sets..."
cp $BASE_CORPUS-dev.en $OUT_DIR/dev.en
cp $BASE_CORPUS-dev.mt $OUT_DIR/dev.mt
cp $BASE_CORPUS-test.en $OUT_DIR/test.en
cp $BASE_CORPUS-test.mt $OUT_DIR/test.mt

echo "Applying BPE tokenization..."
for SPLIT in train dev test; do
    for LANG in en mt; do
        subword-nmt apply-bpe -c $BPE_CODES < $OUT_DIR/${SPLIT}.${LANG} > $OUT_DIR/${SPLIT}.bpe.${LANG}
    done
done

echo "All steps completed. Tokenized files are in $OUT_DIR."

echo "Running Fairseq Preprocessing..."
fairseq-preprocess \
    --source-lang mt \
    --target-lang en \
    --trainpref $OUT_DIR/train.bpe \
    --validpref $OUT_DIR/dev.bpe \
    --testpref $OUT_DIR/test.bpe \
    --destdir $OUT_DIR \
    --srcdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.en.txt \
    --tgtdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.mt.txt \
    --workers 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fairseq-train \
    $OUT_DIR \
    --fp16 \
    --arch transformer \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --clip-norm 0.0 \
    --lr 1e-3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-update 150000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 10 \
    --encoder-layers 6 \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 \
    --decoder-layers 6 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 2048 \
    --decoder-attention-heads 8 \
    --save-dir checkpoints/transformer_mt_en_boosted_$TYPE \
    --log-format json \
    --log-interval 100 \
    --log-file checkpoints/transformer_mt_en_boosted_$TYPE/log_mt_en.json \
    --save-interval-updates 1000 \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --ddp-backend no_c10d \
    --distributed-world-size 4 \
    --distributed-num-procs 4 \
    --update-freq 8