# Linux manipulation
import os
import argparse

# Data manipulation
import pandas as pd
import numpy as np

# Machine learning models and metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Text preprocessing
import string
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Dataset
from datasets import load_dataset

# Define arguments
parser = argparse.ArgumentParser(description='Script for SVM classification with GloVe and PPMI embeddings')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')
parser.add_argument('--ppmi_type', type=str, required=True, help='Type of PPMI space (all / no)')

args = parser.parse_args()

# Define language 
language = args.language
ppmi_type = args.ppmi_type
languages_mapping = {"ro": "romanian", "da": "danish", "he": "hebrew", "sl": "slovenian", "lv": "latvian", "th": "thai", "ur": "urdu", "cy": "welsh", "az": 'azerbaijani', "el": 'greek', "sk": 'slovak', "ka": 'georgian', "bn": 'bengali', "mk": 'macedonian','ku': 'kurdish', 'te': 'telugu', 'mr': 'marathi', 'uz': "uzbek", 'sw': 'swahili', 'yo': 'yoruba', 'ug': "uyghur", 'ne': 'nepali', 'jv': 'javanese', 'si': 'sinhala', 'su': 'sundanese', 'bg': 'bulgarian', 'am': 'amharic'}

# Define embedding paths
glove_path = f"/ds/text/cc100/vectors_more/vector-{language}.txt"

if ppmi_type == "all":
    ppmi_path = "/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn_all/ppmi_embeddings_all.txt"
else:
    ppmi_path = f"/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn/ppmi_embeddings_{language}.txt"

# Load data
dataset = load_dataset(f"DGurgurov/{languages_mapping[language]}_sa")
print('------------> Data loaded!')

# Define functinos
def read_embeddings_from_text(file_path, embedding_size=300):
    """Function to read the embeddings from a txt file"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            # Determine where the embedding starts based on the embedding size
            embedding_start_index = len(parts) - embedding_size
            # Join parts up to the start of the embedding as the phrase
            phrase = ' '.join(parts[:embedding_start_index])
            # Convert the rest to a numpy array of floats as the embedding
            embedding = np.array([float(val) for val in parts[embedding_start_index:]])
            embeddings[phrase] = embedding
    return embeddings

def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

# Load embeddings
glove = read_embeddings_from_text(glove_path)
ppmi = read_embeddings_from_text(ppmi_path)
print('------------> Embeddings loaded!')

# Compute common vocab between emb spaces
common_words = [word for word in glove if word in ppmi and len(glove[word]) == 300]
print(f'Common vocabulary between the embedding spaces for {language}({languages_mapping[language]}):', len(common_words))

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

glove_coverage = len([word for word in combined_vocab if word in glove])
print('Glove length:', len(glove))
print('PPMI length:', len(ppmi))
print(f"Ratio of the known words both in SA and Glove to the total num of SA: {glove_coverage / len(combined_vocab)}%")