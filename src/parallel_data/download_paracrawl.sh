#!/bin/bash

# Define variables
URL="https://object.pouta.csc.fi/OPUS-ParaCrawl/v9/moses/en-mt.txt.zip"
ZIP_FILE="en-mt.txt.zip"
OUTPUT_DIR="en-mt-paracrawl"
EN_FILE="en-mt.en"
MT_FILE="en-mt.mt"
SPLIT_RATIO=(80 10 10) # Train, Dev, Test split ratios
SHUF_SEED=42 # Seed for reproducibility

# Download the zip file
if [ ! -f "$ZIP_FILE" ]; then
  echo "Downloading $ZIP_FILE..."
  wget -q "$URL" -O "$ZIP_FILE"
else
  echo "$ZIP_FILE already exists. Skipping download."
fi

# Unzip the file
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Unzipping $ZIP_FILE..."
  mkdir -p "$OUTPUT_DIR"
  unzip -q "$ZIP_FILE" -d "$OUTPUT_DIR"
else
  echo "$OUTPUT_DIR already exists. Skipping unzipping."
fi

# Check if required files exist
if [ ! -f "$OUTPUT_DIR/ParaCrawl.en-mt.en" ] || [ ! -f "$OUTPUT_DIR/ParaCrawl.en-mt.mt" ]; then
  echo "Error: Required files en-mt.en or en-mt.mt not found."
  exit 1
fi

# Shuffle and split data
# echo "Shuffling and splitting data..."
# paste "$OUTPUT_DIR/ParaCrawl.en-mt.en" "$OUTPUT_DIR/ParaCrawl.en-mt.mt" | shuf --random-source=<(yes $SHUF_SEED) > "$OUTPUT_DIR/shuffled.txt"

# TOTAL_LINES=$(wc -l < "$OUTPUT_DIR/shuffled.txt")
# TRAIN_LINES=$(( TOTAL_LINES * ${SPLIT_RATIO[0]} / 100 ))
# DEV_LINES=$(( TOTAL_LINES * ${SPLIT_RATIO[1]} / 100 ))
# TEST_LINES=$(( TOTAL_LINES - TRAIN_LINES - DEV_LINES ))

# head -n "$TRAIN_LINES" "$OUTPUT_DIR/shuffled.txt" | cut -f1 > "$OUTPUT_DIR/train.en"
# head -n "$TRAIN_LINES" "$OUTPUT_DIR/shuffled.txt" | cut -f2 > "$OUTPUT_DIR/train.mt"

# tail -n +$(( TRAIN_LINES + 1 )) "$OUTPUT_DIR/shuffled.txt" | head -n "$DEV_LINES" | cut -f1 > "$OUTPUT_DIR/dev.en"
# tail -n +$(( TRAIN_LINES + 1 )) "$OUTPUT_DIR/shuffled.txt" | head -n "$DEV_LINES" | cut -f2 > "$OUTPUT_DIR/dev.mt"

# tail -n "$TEST_LINES" "$OUTPUT_DIR/shuffled.txt" | cut -f1 > "$OUTPUT_DIR/test.en"
# tail -n "$TEST_LINES" "$OUTPUT_DIR/shuffled.txt" | cut -f2 > "$OUTPUT_DIR/test.mt"

# # Cleanup
# rm "$OUTPUT_DIR/shuffled.txt"

# echo "Data preparation complete. Files saved in $OUTPUT_DIR:"  
# echo "- train.en / train.mt"
# echo "- dev.en / dev.mt"
# echo "- test.en / test.mt"
