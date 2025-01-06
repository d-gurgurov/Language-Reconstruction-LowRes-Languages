#!/bin/bash

pip install -r requirements.txt

# Define paths
NEWSCRAWL_EN="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/newscrawl/newscrawl.syn.1m.mt"
NEWSCRAWL_MT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/newscrawl/newscrawl.syn.1m.mt"
BPE_CODES="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/codes.bpe"
MODEL_PATH="/netscratch/dgurgurov/projects2024/mt_lrls/src/checkpoints/transformer_mt_mt/checkpoint_best.pt"

# Create output directories
mkdir -p data-bin/reconstruct

# Step 1: Apply BPE tokenization to ParaCrawl English data
echo "Applying BPE tokenization..."
subword-nmt apply-bpe -c $BPE_CODES < $NEWSCRAWL_EN > data-bin/reconstruct/newscrawl.bpe.en
subword-nmt apply-bpe -c $BPE_CODES < $NEWSCRAWL_MT > data-bin/reconstruct/newscrawl.bpe.mt


# Step 2: Create dummy target file for preprocessing
# (fairseq-preprocess requires both source and target files)

# Step 3: Preprocess the data using the same dictionary as training
echo "Preprocessing data..."
fairseq-preprocess \
    --source-lang en --target-lang mt \
    --trainpref data-bin/reconstruct/newscrawl.bpe \
    --destdir data-bin/reconstruct \
    --srcdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.en.txt \
    --tgtdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-mt/dict.mt.txt \
    --workers 20

OUTPUT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/reconstruct"

# Step 4: Generate translations
echo "Generating translations..."
fairseq-generate $OUTPUT \
    --fp16 \
    --path $MODEL_PATH \
    --source-lang en --target-lang mt \
    --beam 4 \
    --gen-subset train \
    --lenpen 0.6 \
    --batch-size 1 \
    --sacrebleu \
    --remove-bpe \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --max-tokens 4096 \
    --results-path $OUTPUT

# Remove BPE - optional - if not specified in the command above
# sed -r 's/ ?@@ ?//g' data-bin/newscrawl/newscrawl.out > data-bin/newscrawl/newscrawl.out


grep ^H $OUTPUT/generate-train.txt | cut -f 3 | sacremoses detokenize > $OUTPUT/gen.detok.out
grep ^T $OUTPUT/generate-train.txt | cut -f 2 | sacremoses detokenize > $OUTPUT/gen.ref.detok
echo "SacreBLEU score one line below..."
sacrebleu $OUTPUT/gen.ref.detok -i $OUTPUT/gen.detok.out -m bleu -b -w 4


# Extract hypothesis lines, sort by ID, and remove the scores
# grep "^H-" $INPUT_FILE | sort -t'-' -k2,2n | cut -f3- > $OUTPUT_FILE

# Print some statistics
echo "Statistics:"
echo "Original English sentences: $(wc -l < $NEWSCRAWL_EN)"
echo "Generated Maltese sentences: $(wc -l < $OUTPUT/generate-train.txt)"