# Linux
import os
import argparse

# Data manipulation
import pandas as pd
import numpy as np

# Machine learning models and metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Undersampling
from imblearn.under_sampling import RandomUnderSampler

# Hugging Face Transformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

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
parser.add_argument('--kernel', type=str, required=True, help='Kernel type for the SVM (e.g., linear, rbf)')

args = parser.parse_args()

# Define language 
language = args.language
kernel_type = args.kernel
languages_mapping = {"ro": "romanian", "da": "danish", "he": "hebrew", 
                     "sl": "slovenian", "lv": "latvian", "th": "thai", 
                     "ur": "urdu", "cy": "welsh", "az": 'azerbaijani', 
                     "el": 'greek', "sk": 'slovak', "ka": 'georgian', 
                     "bn": 'bengali', "mk": 'macedonian','ku': 'kurdish', 
                     'te': 'telugu', 'mr': 'marathi', 'uz': "uzbek", 
                     'sw': 'swahili', 'yo': 'yoruba', 'ug': "uyghur", 
                     'ne': 'nepali', 'jv': 'javanese', 'si': 'sinhala', 
                     'su': 'sundanese', 'bg': 'bulgarian', 'am': 'amharic'}

# Load dataset
dataset = load_dataset(f"DGurgurov/{languages_mapping[language]}_sa")

# Define function for text preprocessing
def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

# Prepare data
train = pd.DataFrame(dataset["train"])
valid = pd.DataFrame(dataset["validation"])
test = pd.DataFrame(dataset["test"])

train['text'] = train['text'].apply(preprocess_text)
valid['text'] = valid['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

X_train = train['text']
y_train = train['label']

X_valid = valid['text']
y_valid = valid['label']

X_test = test['text']
y_test = test['label']

# Load XLM-R model and tokenizer
xlmr_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base').to('cuda')
xlmr_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

# Load E5 model
e5_model = SentenceTransformer('intfloat/multilingual-e5-base')

# Function to embed a single sentence using XLM-R
def embed_sentence(sentence):
    encoded_input = xlmr_tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = xlmr_model(**encoded_input)
    embedding = model_output.last_hidden_state.sum(dim=1).cpu().numpy()
    return embedding

# Function to embed all sentences individually and aggregate into a matrix
def embed_sentences(texts):
    embeddings = []
    for text in texts:
        embedding = embed_sentence(text)
        embeddings.append(embedding)
    return np.vstack(embeddings)


# Function to embed a single sentence using e5-base
def embed_sentences_e5(texts):
    prefixed_texts = ["query: " + text for text in texts]
    embeddings = e5_model.encode(prefixed_texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# Embed sentences
X_train_embeddings = embed_sentences_e5(X_train.tolist())
X_valid_embeddings = embed_sentences_e5(X_valid.tolist())
X_test_embeddings = embed_sentences_e5(X_test.tolist())

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [100], 'kernel': ['rbf']}

def hyperparam_tuning(xtrain, ytrain, xvalid, yvalid, classifier, param_grid):
    """Function that performs the hyperparameter search"""
    """
    If hyperparameters need to be tuned, use:
    split_index = [-1] * len(xtrain) + [0] * len(xvalid)
    ps = PredefinedSplit(split_index)

    x_combined = np.vstack((xtrain, xvalid))
    y_combined = np.concatenate((ytrain, yvalid))

    clf = GridSearchCV(classifier, param_grid, cv=ps)
    clf.fit(x_combined, y_combined)
    
    We fix the hyperparameters:
    """
    clf = GridSearchCV(classifier, param_grid)
    print(xtrain.shape)
    print(ytrain.shape)
    clf.fit(xtrain, ytrain)

    best_params = clf.best_params_
    print('Hyperparameters:', best_params)

    final_model = classifier.set_params(**best_params)
    final_model.fit(xtrain, ytrain)

    return final_model

# Train SVM model
svm_classifier = SVC()
svm_model = hyperparam_tuning(X_train_embeddings, y_train, X_valid_embeddings, y_valid, svm_classifier, param_grid)

# Evaluate the model
test_predictions = svm_model.predict(X_test_embeddings)
print("SVM with XLM-R embeddings:")
print(classification_report(y_test, test_predictions, digits=3))

def save_results(file_path, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(f"Accuracy: {accuracy:.5f}\n")
        file.write(f"Macro Average F1 Score: {macro_f1:.5f}\n")
        file.write(f"Micro Average F1 Score: {micro_f1:.5f}\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=', '))

save_results(f'./e5_base_rbf_100/{language}/svm_e5.txt', y_test, test_predictions)
