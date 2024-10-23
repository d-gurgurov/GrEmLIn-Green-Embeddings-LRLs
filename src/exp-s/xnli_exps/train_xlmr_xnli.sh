#!/bin/bash

pip install imblearn nltk
pip install sentence_transformers~=2.2.2

# List of languages
all_languages=("sw" "bg" "th" "ur" "el")

# Kernel type
kernel="rbf"


# Iterate through each language and run the Python script
for language in "${all_languages[@]}"; do
    echo "Running for language: $language with kernel: $kernel"
    python xlmr_xnli.py --language "$language" --kernel "$kernel"
done