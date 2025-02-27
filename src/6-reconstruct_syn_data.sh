#!/bin/bash

pip install -r requirements.txt

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install fairseq==0.12.2

LANGUAGE="sl"

# Define paths
NEWSCRAWL_EN="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/newscrawl_$LANGUAGE/newscrawl.syn.1m.$LANGUAGE"
NEWSCRAWL_MT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/newscrawl_$LANGUAGE/newscrawl.syn.1m.$LANGUAGE"
BPE_CODES="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-$LANGUAGE/codes.bpe"
MODEL_PATH="/netscratch/dgurgurov/projects2024/mt_lrls/src/checkpoints/transformer_${LANGUAGE}_${LANGUAGE}/checkpoint_best.pt"

# Create output directories
mkdir -p data-bin/reconstruct_$LANGUAGE

# Step 1: Apply BPE tokenization to ParaCrawl English data
echo "Applying BPE tokenization..."
subword-nmt apply-bpe -c $BPE_CODES < $NEWSCRAWL_EN > data-bin/reconstruct_$LANGUAGE/newscrawl.bpe.en
subword-nmt apply-bpe -c $BPE_CODES < $NEWSCRAWL_MT > data-bin/reconstruct_$LANGUAGE/newscrawl.bpe.$LANGUAGE


# Step 2: Create dummy target file for preprocessing
# (fairseq-preprocess requires both source and target files)

# Step 3: Preprocess the data using the same dictionary as training
echo "Preprocessing data..."
fairseq-preprocess \
    --source-lang en --target-lang $LANGUAGE \
    --trainpref data-bin/reconstruct_$LANGUAGE/newscrawl.bpe \
    --destdir data-bin/reconstruct_$LANGUAGE \
    --srcdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-$LANGUAGE/dict.en.txt \
    --tgtdict /netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/en-$LANGUAGE/dict.$LANGUAGE.txt \
    --workers 20

OUTPUT="/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/reconstruct_$LANGUAGE"

# Step 4: Generate translations
echo "Generating translations..."
fairseq-generate $OUTPUT \
    --fp16 \
    --path $MODEL_PATH \
    --source-lang en --target-lang $LANGUAGE \
    --beam 4 \
    --gen-subset train \
    --lenpen 0.6 \
    --batch-size 128 \
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
echo "Generated Target sentences: $(wc -l < $OUTPUT/generate-train.txt)"
