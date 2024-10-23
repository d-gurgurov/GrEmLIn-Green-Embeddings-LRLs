#!/bin/bash 

# download ConceptNet data from the official website
# file name - conceptnet-assertions-5.7.0.csv.gz
# link - https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz


echo "Running cn_parser.py..."
python cn_parser.py

echo "Running cn_inspect.py..."
python cn_inspect.py

echo "Running cn_clean.py..."
python cn_clean.py

echo "Running cn_analyze.py..."
python cn_analyze.py


# output files:
# 1. complete conceptnet data without sources and relation types
# 2. clean concpenete data without prefixes and underscores between words
# 3. analysis file with the number of languages and edges for each language