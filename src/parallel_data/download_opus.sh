#!/bin/bash

# Create directories for each language pair
mkdir -p en-mt en-si en-ug

# Download English-Maltese files
cd en-mt
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-mt/opus.en-mt-dev.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-mt/opus.en-mt-test.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-mt/opus.en-mt-train.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-mt/opus.en-mt-dev.mt
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-mt/opus.en-mt-test.mt
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-mt/opus.en-mt-train.mt
cd ..

# Download English-Slovenian files
cd en-si
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-si/opus.en-si-dev.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-si/opus.en-si-test.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-si/opus.en-si-train.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-si/opus.en-si-dev.si
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-si/opus.en-si-test.si
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-si/opus.en-si-train.si
cd ..

# Download English-Ugandan files
cd en-ug
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ug/opus.en-ug-dev.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ug/opus.en-ug-test.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ug/opus.en-ug-train.en
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ug/opus.en-ug-dev.ug
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ug/opus.en-ug-test.ug
wget https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ug/opus.en-ug-train.ug
cd ..

# Unzip files if they are compressed (uncomment if needed)
# cd en-mt && unzip '*.zip' && cd ..
# cd en-si && unzip '*.zip' && cd ..
# cd en-ug && unzip '*.zip' && cd ..

echo "Download complete!"