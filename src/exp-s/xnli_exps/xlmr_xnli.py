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
import torch

# Text preprocessing
import string
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Define arguments
parser = argparse.ArgumentParser(description='Script for SVM classification with GloVe and PPMI embeddings')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')
parser.add_argument('--kernel', type=str, required=True, help='Kernel type for the SVM (e.g., linear, rbf)')

args = parser.parse_args()

# Define language 
language = args.language
kernel_type = args.kernel

# Load XLM-R model and tokenizer
xlmr_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base').to('cuda')
xlmr_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

# Load E5 model
e5_model = SentenceTransformer('intfloat/multilingual-e5-base')

# Function to embed a single sentence using XLM-R
def embed_sentence(sentence):
    encoded_input = xlmr_tokenizer(sentence, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = xlmr_model(**encoded_input)
    embedding = model_output.last_hidden_state.sum(dim=1).cpu().numpy()
    return embedding

# Function to embed a single sentence using e5-base
def embed_sentence_e5(sentence):
    prefixed_text = "query: " + sentence
    embedding = e5_model.encode([prefixed_text], convert_to_tensor=True)
    return embedding.cpu().numpy()[0]  # Return the first (and only) embedding

# Load dataset (TSV format with columns: gold_label, language, sentence1, sentence2)
xnli_data = pd.read_csv('./data/xnli.dev.tsv', sep='\t')
xnli_data_test = pd.read_csv('./data/xnli.test.tsv', sep='\t')
language = args.language
xnli_data = xnli_data[xnli_data['language'] == language]
xnli_data_test = xnli_data_test[xnli_data_test['language'] == language]

# Define function for text preprocessing
def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

xnli_data['sentence1'] = xnli_data['sentence1'].apply(preprocess_text)
xnli_data_test['sentence1'] = xnli_data_test['sentence1'].apply(preprocess_text)
xnli_data['sentence2'] = xnli_data['sentence2'].apply(preprocess_text)
xnli_data_test['sentence2'] = xnli_data_test['sentence2'].apply(preprocess_text)

def embed_sentence_pair(row):
    # Get the embeddings for both sentences
    sentence1_embedding = embed_sentence_e5(row['sentence1'])
    sentence2_embedding = embed_sentence_e5(row['sentence2'])
    
    # Concatenate them to form a single vector per sentence pair
    combined_embedding = np.concatenate((sentence1_embedding, sentence2_embedding), axis=0)  # Correct concatenation along columns
    
    return combined_embedding

# Apply the function and store embeddings in a list
train_embeddings = []
for _, row in xnli_data.iterrows():
    train_embeddings.append(embed_sentence_pair(row))

# Stack the embeddings and check the shape
X_train = np.vstack(train_embeddings)
print("X_train shape:", X_train.shape)

# Ensure y_train matches the length of X_train
y_train = xnli_data['gold_label'].values
print("y_train shape:", y_train.shape)

# Repeat for the test set
test_embeddings = []
for _, row in xnli_data_test.iterrows():
    test_embeddings.append(embed_sentence_pair(row))

X_test = np.vstack(test_embeddings)
print("X_test shape:", X_test.shape)

y_test = xnli_data_test['gold_label'].values
print("y_test shape:", y_test.shape)


# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [100], 'kernel': ['rbf']}

def hyperparam_tuning(xtrain, ytrain, classifier, param_grid):
    """Function that performs the hyperparameter search"""
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
svm_model = hyperparam_tuning(X_train, y_train, svm_classifier, param_grid)

# Evaluate the model
test_predictions = svm_model.predict(X_test)
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

save_results(f'./e5_base_rbf_100/{language}/svm.txt', y_test, test_predictions)
