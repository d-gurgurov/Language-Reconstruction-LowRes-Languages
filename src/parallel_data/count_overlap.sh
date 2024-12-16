#!/bin/bash

# Define file paths
file1="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt/opus.en-mt-train.en"
file2="/netscratch/dgurgurov/projects2024/mt_lrls/parallel_data_mt/en-mt-paracrawl/ParaCrawl.en-mt.en"

# Create temporary sorted files for faster processing
echo "Sorting files..."
sort "$file1" | uniq > temp1.txt
sort "$file2" | uniq > temp2.txt

# Count lines in each file
total1=$(wc -l < temp1.txt)
total2=$(wc -l < temp2.txt)

# Find common lines
comm -12 temp1.txt temp2.txt > common.txt
overlap=$(wc -l < common.txt)

# Calculate percentages
awk -v overlap="$overlap" -v total1="$total1" -v total2="$total2" 'BEGIN {
    printf "Total lines in file1: %d\n", total1
    printf "Total lines in file2: %d\n", total2
    printf "Overlapping lines: %d\n", overlap
    printf "Percentage of file1 that overlaps: %.2f%%\n", (overlap/total1)*100
    printf "Percentage of file2 that overlaps: %.2f%%\n", (overlap/total2)*100
}'

# Clean up temporary files
rm temp1.txt temp2.txt common.txt