#!/bin/bash

pip install nltk fasttext

# List of languages
languages=("bg" "el" "sw" "th" "ur")

# Iterate through each language and run the Python script
for language in "${languages[@]}"; do
    echo "Running for language: $language"
    python embs_stats_xnli.py --language "$language"
done