#!/bin/bash

pip install imblearn

# List of languages
all_languages=("si" "am" "su" "sw" "ka" "ne" "ug" "yo" "ur" "mk" "mr" "bn" "te" "uz" "az" "sl" "lv" "ro" "he" "cy" "bg" "sk" "da")

imbalanced_languages=("sw" "ne" "ug" "lv" "sk" "sl" "uz" "bg" "yo" "bn" "he" "te")

# Kernel type
kernel="rbf"

# Alignment type
alignment="svd"

# Iterate through each language and run the Python script
for language in "${imbalanced_languages[@]}"; do
    echo "Running for language: $language with kernel: $kernel"
    python retrofit_project.py --language "$language" --kernel "$kernel" --align "$alignment"
done