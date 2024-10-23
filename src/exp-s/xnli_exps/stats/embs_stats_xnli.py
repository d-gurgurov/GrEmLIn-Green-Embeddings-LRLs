import os
import argparse
import pandas as pd
import numpy as np
import string
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import fasttext

# Define arguments
parser = argparse.ArgumentParser(description='Script for FastText and GloVe embedding coverage analysis on XNLI dataset')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')

args = parser.parse_args()

# Define language 
language = args.language
languages_mapping = {"ro": "romanian", "da": "danish", "he": "hebrew", "sl": "slovenian", "lv": "latvian", "th": "thai", "ur": "urdu", "cy": "welsh", "az": 'azerbaijani', "el": 'greek', "sk": 'slovak', "ka": 'georgian', "bn": 'bengali', "mk": 'macedonian','ku': 'kurdish', 'te': 'telugu', 'mr': 'marathi', 'uz': "uzbek", 'sw': 'swahili', 'yo': 'yoruba', 'ug': "uyghur", 'ne': 'nepali', 'jv': 'javanese', 'si': 'sinhala', 'su': 'sundanese', 'bg': 'bulgarian', 'am': 'amharic'}

# Define embedding paths
fasttext_path = f"/ds/text/cc100/fasttext/cc.{language}.300.bin"
glove_path = f"/ds/text/cc100/vectors_more/vector-{language}.txt"

# Load XNLI dataset
xnli_data = pd.read_csv('./data/xnli.dev.tsv', sep='\t')
xnli_data_test = pd.read_csv('./data/xnli.test.tsv', sep='\t')

# Filter data for the specified language
xnli_data = xnli_data[xnli_data['language'] == language]
xnli_data_test = xnli_data_test[xnli_data_test['language'] == language]

print(f'------------> Data loaded for language: {language}')

def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

# Preparing data
xnli_data['sentence1'] = xnli_data['sentence1'].apply(preprocess_text)
xnli_data['sentence2'] = xnli_data['sentence2'].apply(preprocess_text)
xnli_data_test['sentence1'] = xnli_data_test['sentence1'].apply(preprocess_text)
xnli_data_test['sentence2'] = xnli_data_test['sentence2'].apply(preprocess_text)

# Create vocabulary from sentence1 and sentence2
def build_vocabulary(dataframe):
    vocabulary = set()
    for _, row in dataframe.iterrows():
        words1 = row['sentence1'].split()
        words2 = row['sentence2'].split()
        vocabulary.update(words1)
        vocabulary.update(words2)
    return vocabulary

dev_vocab = build_vocabulary(xnli_data)
test_vocab = build_vocabulary(xnli_data_test)

# Combine dev and test vocabulary
combined_vocab = dev_vocab.union(test_vocab)

# Load FastText model
print("Loading FastText model...")
fasttext_model = fasttext.load_model(fasttext_path)
print("FastText model loaded!")

# Load GloVe embeddings
def read_embeddings_from_text(file_path, embedding_size=300):
    """Function to read the embeddings from a txt file"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            parts = line.strip().split()
            if len(parts) <= embedding_size:
                print(f"Warning: Line {line_num} does not have enough values. Skipping.")
                continue
            
            # Determine where the embedding starts based on the embedding size
            embedding_start_index = len(parts) - embedding_size
            
            # Join parts up to the start of the embedding as the phrase
            phrase = ' '.join(parts[:embedding_start_index])
            
            try:
                # Convert the rest to a numpy array of floats as the embedding
                embedding = np.array([float(val) for val in parts[embedding_start_index:]])
                
                if len(embedding) != embedding_size:
                    print(f"Warning: Embedding for '{phrase}' on line {line_num} has {len(embedding)} dimensions instead of {embedding_size}. Skipping.")
                    continue
                
                embeddings[phrase] = embedding
            except ValueError as e:
                print(f"Error on line {line_num}: Could not convert embedding to float for word '{phrase}'. Skipping.")
                print(f"Problematic values: {parts[embedding_start_index:]}")
                continue
    
    print(f"Loaded {len(embeddings)} embeddings.")
    return embeddings

print("Loading GloVe embeddings...")
glove = read_embeddings_from_text(glove_path)
print("GloVe embeddings loaded!")

# Function to check if FastText provides a non-zero embedding for a word
def has_non_zero_embedding(word, model):
    embedding = model.get_word_vector(word)
    return not np.allclose(embedding, 0)

# Compute FastText coverage
fasttext_coverage = sum(1 for word in combined_vocab if has_non_zero_embedding(word, fasttext_model))
total_words = len(combined_vocab)

fasttext_coverage_ratio = fasttext_coverage / total_words

# Compute GloVe coverage
glove_coverage = sum(1 for word in combined_vocab if word in glove)
glove_coverage_ratio = glove_coverage / total_words

print(f"Total vocabulary size: {total_words}")
print("\nFastText Coverage:")
print(f"Words with non-zero FastText embeddings: {fasttext_coverage}")
print(f"Ratio of words with non-zero FastText embeddings: {fasttext_coverage_ratio:.2%}")
print(f"Ratio of words with zero FastText embeddings: {1 - fasttext_coverage_ratio:.2%}")

print("\nGloVe Coverage:")
print(f"Words with GloVe embeddings: {glove_coverage}")
print(f"Ratio of words with GloVe embeddings: {glove_coverage_ratio:.2%}")
print(f"Ratio of words without GloVe embeddings: {1 - glove_coverage_ratio:.2%}")

print("\nComparison:")
print(f"FastText coverage is {(fasttext_coverage_ratio - glove_coverage_ratio) * 100:.2f} percentage points {'higher' if fasttext_coverage_ratio > glove_coverage_ratio else 'lower'} than GloVe coverage.")