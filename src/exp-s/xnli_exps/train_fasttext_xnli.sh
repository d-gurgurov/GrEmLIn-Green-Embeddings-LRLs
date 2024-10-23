#!/bin/bash

pip install imblearn nltk fasttext

# List of languages
all_languages=("bg" "el" "sw" "th" "ur")

# Kernel type
kernel="rbf"


# Iterate through each language and run the Python script
for language in "${all_languages[@]}"; do
    echo "Running for language: $language with kernel: $kernel"
    python test_fasttext.py --language "$language" --kernel "$kernel"
done