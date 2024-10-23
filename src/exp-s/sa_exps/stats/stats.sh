#!/bin/bash

pip install imblearn nltk

# List of languages
languages=("am" "su" "sw" "ka" "ne" "ug" "yo" "ur" "mk" "mr" "bn" "te" "uz" "az" "sl" "lv" "ro" "he" "cy" "bg" "sk" "da" "si")

# PPMI emb space
ppmi_space="single"

# Iterate through each language
for language in "${languages[@]}"; do
    echo "Running for language: $language with kernel: $kernel"
    python embs_stats.py --language "$language" --ppmi_type "$ppmi_space"
done