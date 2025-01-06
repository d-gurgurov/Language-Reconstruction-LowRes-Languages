#!/bin/bash

# Input file containing Fairseq output
INPUT_FILE="generate-train.txt"

# Output file for reordered hypothesis sentences
OUTPUT_FILE="reordered.newscrawl.boost.1m.mt"

echo "Extracting and reordering hypothesis lines..."

# Extract hypothesis lines, sort by ID, and remove the scores
grep "^H-" $INPUT_FILE | sort -t'-' -k2,2n | cut -f3- > $OUTPUT_FILE

echo "Reordered hypotheses saved to $OUTPUT_FILE."
