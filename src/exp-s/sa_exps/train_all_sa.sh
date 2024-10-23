#!/bin/bash

pip install imblearn

# Kernel type
kernel="rbf"

# Alignment type
alignment="svd"

# Projections on to the full PPMI embedding space; langs are defined within the script
python retrofit_project_all.py --kernel "$kernel" --align "$alignment"