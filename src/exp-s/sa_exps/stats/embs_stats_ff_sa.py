import os
import argparse
import pandas as pd
import numpy as np
import string
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from datasets import load_dataset
import fasttext

# Define arguments
parser = argparse.ArgumentParser(description='Script for FastText embedding coverage analysis')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')

args = parser.parse_args()

# Define language 
language = args.language
languages_mapping = {"ro": "romanian", "da": "danish", "he": "hebrew", "sl": "slovenian", "lv": "latvian", "th": "thai", "ur": "urdu", "cy": "welsh", "az": 'azerbaijani', "el": 'greek', "sk": 'slovak', "ka": 'georgian', "bn": 'bengali', "mk": 'macedonian','ku': 'kurdish', 'te': 'telugu', 'mr': 'marathi', 'uz': "uzbek", 'sw': 'swahili', 'yo': 'yoruba', 'ug': "uyghur", 'ne': 'nepali', 'jv': 'javanese', 'si': 'sinhala', 'su': 'sundanese', 'bg': 'bulgarian', 'am': 'amharic'}

# Define embedding path
fasttext_path = f"/ds/text/cc100/fasttext/cc.{language}.300.bin"

# Load data
dataset = load_dataset(f"DGurgurov/{languages_mapping[language]}_sa")
print('------------> Data loaded!')

def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

# Preparing data
train = pd.DataFrame(dataset["train"])
valid = pd.DataFrame(dataset["validation"])
test = pd.DataFrame(dataset["test"])
train['text'] = train['text'].apply(preprocess_text)
valid['text'] = valid['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

# Create vocabulary from train and test texts
def build_vocabulary(text_series):
    vocabulary = set()
    for text in text_series:
        words = text.split()
        vocabulary.update(words)
    return vocabulary

train_vocab = build_vocabulary(train['text'])
test_vocab = build_vocabulary(test['text'])

# Combine train and test vocabulary
combined_vocab = train_vocab.union(test_vocab)

# Load FastText model
print("Loading FastText model...")
fasttext_model = fasttext.load_model(fasttext_path)
print("FastText model loaded!")

# Function to check if FastText provides a non-zero embedding for a word
def has_non_zero_embedding(word, model):
    embedding = model.get_word_vector(word)
    return not np.allclose(embedding, 0)

# Compute coverage
fasttext_coverage = sum(1 for word in combined_vocab if has_non_zero_embedding(word, fasttext_model))
total_words = len(combined_vocab)

coverage_ratio = fasttext_coverage / total_words

print(f"Total vocabulary size: {total_words}")
print(f"Words with non-zero FastText embeddings: {fasttext_coverage}")
print(f"Ratio of words with non-zero FastText embeddings: {coverage_ratio:.2%}")
print(f"Ratio of words with zero FastText embeddings: {1 - coverage_ratio:.2%}")