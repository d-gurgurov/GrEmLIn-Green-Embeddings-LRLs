#!/bin/bash

pip install imblearn nltk
pip install sentence_transformers~=2.2.2

# List of languages  
all_languages=("si" "am" "su" "sw" "ka" "ne" "ug" "mr" "bn" "te" "uz" "az" "sl" "lv" "ro" "he" "cy" "bg" "sk" "da" "yo" "ur" "mk")

# Kernel type
kernel="rbf"


# Iterate through each language and run the Python script
for language in "${all_languages[@]}"; do
    echo "Running for language: $language with kernel: $kernel"
    python xlmr_sa.py --language "$language" --kernel "$kernel"
done