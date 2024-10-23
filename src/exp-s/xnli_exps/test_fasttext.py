import os
import argparse
import pandas as pd
import numpy as np
import string
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from nltk import word_tokenize
import nltk
import fasttext
nltk.download('punkt')
nltk.download('punkt_tab')

# Define arguments
parser = argparse.ArgumentParser(description='Script for SVM classification with GloVe, FastText, and retrofitted GloVe embeddings on XNLI')
parser.add_argument('--kernel', type=str, required=True, help='Kernel type for the SVM (e.g., linear, rbf)')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')
args = parser.parse_args()

# Load dataset (TSV format with columns: gold_label, language, sentence1, sentence2)
xnli_data = pd.read_csv('./data/xnli.dev.tsv', sep='\t')
xnli_data_test = pd.read_csv('./data/xnli.test.tsv', sep='\t')
language = args.language
xnli_data = xnli_data[xnli_data['language'] == language]
xnli_data_test = xnli_data_test[xnli_data_test['language'] == language]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)

# Define embedding functions
def embed_sentence(sentence, embedding_model):
    words = preprocess_text(sentence)
    embedding = np.zeros(embedding_model.get_dimension())
    for word in words:
        embedding += embedding_model.get_word_vector(word)
    return embedding

# Load embeddings based on user input

model_path = f'/ds/text/cc100/fasttext/cc.{language}.300.bin'
embedding_model = fasttext.load_model(model_path)

# Embed sentence pairs and combine them
def embed_sentence_pair(row, embedding_model):
    sentence1_embedding = embed_sentence(row['sentence1'], embedding_model)
    sentence2_embedding = embed_sentence(row['sentence2'], embedding_model)
    return np.concatenate((sentence1_embedding, sentence2_embedding))

# Prepare data
X_train = np.vstack(xnli_data.apply(lambda row: embed_sentence_pair(row, embedding_model), axis=1))
y_train = xnli_data['gold_label']

X_test = np.vstack(xnli_data_test.apply(lambda row: embed_sentence_pair(row, embedding_model), axis=1))
y_test = xnli_data_test['gold_label']

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [1], 'kernel': ['rbf']}

def hyperparam_tuning(xtrain, ytrain, classifier, param_grid):
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
svm_model = hyperparam_tuning(X_train, y_train, svm_classifier, param_grid)

# Test the model
test_predictions = svm_model.predict(X_test)
print(f"SVM with FastText embeddings:")
print(classification_report(y_test, test_predictions))

# Save results
def save_results(file_path, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(f"Accuracy: {accuracy:.5f}\n")
        file.write(f"Macro F1: {macro_f1:.5f}\n")
        file.write(f"Micro F1: {micro_f1:.5f}\n")
        file.write(f"Confusion Matrix:\n{conf_matrix}\n")

save_results(f'./fasttext/svm_fasttext_{language}.txt', y_test, test_predictions)
