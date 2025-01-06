#!/bin/bash

pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define paths
BPE_CODES="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe"
INPUT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/paracrawl"
OUTPUT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/mt-rec"
SEED=42 

# Create output directory
mkdir -p $OUTPUT

# Define file paths
SYNTHETIC_TRANS="$INPUT/gen.detok.out"
REFERENCE_TRANS="$INPUT/gen.ref.detok"
TRAIN_SYNTH="$OUTPUT/train.detok.en"
TRAIN_REF="$OUTPUT/train.detok.mt"
VALID_SYNTH="$OUTPUT/valid.detok.en"
VALID_REF="$OUTPUT/valid.detok.mt"


# Combine the data into a single file
echo "Combining data..."
paste $SYNTHETIC_TRANS $REFERENCE_TRANS > $OUTPUT/combined_data.txt

# Use Python for reproducible random splitting
echo "Splitting data with seed $SEED..."
python - <<EOF
import random

# Set seed for reproducibility
random.seed($SEED)

# Read combined data
with open("$OUTPUT/combined_data.txt", "r") as f:
    lines = f.readlines()

# Shuffle data
random.shuffle(lines)

# Split into validation and training sets
validation_lines = lines[:2000]
training_lines = lines[2000:]

# Write validation and training splits
with open("$OUTPUT/validation.txt", "w") as val_file:
    val_file.writelines(validation_lines)

with open("$OUTPUT/training.txt", "w") as train_file:
    train_file.writelines(training_lines)
EOF

# Separate synthetic translations and references
cut -f1 $OUTPUT/validation.txt > $VALID_SYNTH
cut -f2 $OUTPUT/validation.txt > $VALID_REF
cut -f1 $OUTPUT/training.txt > $TRAIN_SYNTH
cut -f2 $OUTPUT/training.txt > $TRAIN_REF

echo "Validation split created with 2000 lines. Remaining data used for training."


# Apply BPE to splits
echo "Applying BPE tokenization..."
subword-nmt apply-bpe -c $BPE_CODES < $OUTPUT/train.detok.en > $OUTPUT/train.bpe.en
subword-nmt apply-bpe -c $BPE_CODES < $OUTPUT/train.detok.mt > $OUTPUT/train.bpe.mt
subword-nmt apply-bpe -c $BPE_CODES < $OUTPUT/valid.detok.en > $OUTPUT/valid.bpe.en
subword-nmt apply-bpe -c $BPE_CODES < $OUTPUT/valid.detok.mt > $OUTPUT/valid.bpe.mt

# Preprocess the data
fairseq-preprocess \
    --source-lang en --target-lang mt \
    --trainpref $OUTPUT/train.bpe \
    --validpref $OUTPUT/valid.bpe \
    --destdir data-bin/mt-rec \
    --srcdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.en.txt \
    --tgtdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.mt.txt \
    --workers 20

# Train the model
fairseq-train data-bin/mt-rec \
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
    --patience 10 \
    --encoder-layers 6 \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 \
    --decoder-layers 6 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 2048 \
    --decoder-attention-heads 8 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --save-dir checkpoints/transformer_mt_mt \
    --log-format json \
    --log-interval 100 \
    --log-file checkpoints/transformer_mt_mt/log_mt_mt.json \
    --save-interval-updates 1000 \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --ddp-backend no_c10d \
    --distributed-world-size 4 \
    --distributed-num-procs 4 \
    --update-freq 8
