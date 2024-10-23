#!/bin/bash

pip install imblearn nltk

# List of languages
all_languages=("th" "sw" "el" "bg" "ur")

# Kernel type
kernel="rbf"

# Alignment type
alignment="svd"

# Iterate through each language and run the Python script
for language in "${all_languages[@]}"; do
    echo "Running for language: $language with kernel: $kernel"
    python retrofit_project.py --language "$language" --kernel "$kernel" --align "$alignment"
done