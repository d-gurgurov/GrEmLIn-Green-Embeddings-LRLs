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
from sklearn.preprocessing import MinMaxScaler
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

# Define functinos
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

def sentence_to_embedding(sentence, word_embeddings):
    """Function that converts a sentence into an embedding by combining word embeddings"""
    if len(sentence)==0: 
        return np.zeros(300)
    words = sentence.split()
    embeddings = [word_embeddings.get(word, np.zeros(300)) for word in words]
    return np.sum(embeddings, axis=0)

def align_embeddings_transform_matrix(embeddings_dict_1, embeddings_dict_2):
    """Function to align and retrofit embeddings from different sources by finding a transformation matrix
    between the vector spaces and then applying this matrix to all the embeddings to be retrofitted"""
    common_words = [word for word in embeddings_dict_1 if word in embeddings_dict_2 and len(embeddings_dict_1[word]) == 300]

    print('Common vocabulary between the embedding spaces:', len(common_words))

    model_1 = np.array([embeddings_dict_1[word] for word in common_words])
    model_2 = np.array([embeddings_dict_2[word] for word in common_words])
    
    model = LinearRegression()  
    model.fit(model_1, model_2)
    transformation_matrix = model.coef_.T

    transformed_embeddings = {word: np.dot(embeddings_dict_1[word], transformation_matrix) for word in embeddings_dict_1}

    return transformed_embeddings 

def cca_alignment(embeddings_dict_1, embeddings_dict_2):
    """Function to align embeddings using CCA"""
    common_words = [word for word in embeddings_dict_1 if word in embeddings_dict_2 and len(embeddings_dict_1[word]) == 300]

    print('Common vocabulary between the embedding spaces:', len(common_words))

    # Prepare data for CCA
    embeddings_1 = np.array([embeddings_dict_1[word] for word in common_words])
    embeddings_2 = np.array([embeddings_dict_2[word] for word in common_words])

    # Perform CCA
    cca = CCA(n_components=300, scale=True)
    cca.fit(embeddings_1, embeddings_2)
    transformation = cca.coef_.T

    # Project embeddings using CCA
    aligned_embeddings_1 = {word: np.dot(embeddings_dict_1[word], transformation) for word in embeddings_dict_1}
    aligned_embeddings_2 =  {word: np.dot(embeddings_dict_2[word], transformation) for word in embeddings_dict_2}

    return aligned_embeddings_1, aligned_embeddings_2

def svd_alignment(embeddings_dict_1, embeddings_dict_2):
    """Function to align embeddings using SVD"""
    common_words = [word for word in embeddings_dict_1 if word in embeddings_dict_2 and len(embeddings_dict_1[word]) == 300]

    print('Common vocabulary between the embedding spaces:', len(common_words))

    # Prepare data for CCA
    embeddings_1 = np.array([embeddings_dict_1[word] for word in common_words])
    minimum = np.min(embeddings_1)
    maximum = np.max(embeddings_1)

    # Normalizing PPMI
    embeddings_2 = np.array([embeddings_dict_2[word] for word in common_words])
    embeddings_2 = normalize_embeddings(embeddings_2, minimum, maximum)

    # Concat common vocab
    concatenated_embeddings = np.concatenate((embeddings_1, embeddings_2), axis=1)

    # Perfrom SVD on concatenated embeddings
    U, S, Vt = np.linalg.svd(concatenated_embeddings, full_matrices=False)
    reduced_embeddings = U[:, :300]
    

    # Use Linear Regression to find transformation matrix from GloVe to PCA-reduced space
    model = LinearRegression()
    model.fit(embeddings_1, reduced_embeddings)
    transformation = model.coef_.T

    # Transform the GloVe embeddings
    transformed_embeddings = {word: np.dot(embeddings_dict_1[word], transformation) for word in embeddings_dict_1}

    return transformed_embeddings

def normalize_embeddings(embeddings, min, max):
    """Normalize embeddings to be in the range from -1 to 1."""
    scaler = MinMaxScaler(feature_range=(min, max))
    return scaler.fit_transform(embeddings)

def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

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
    clf.fit(xtrain, ytrain)

    best_params = clf.best_params_
    print('Hyperparameters:', best_params)

    final_model = classifier.set_params(**best_params)
    final_model.fit(xtrain, ytrain)

    return final_model

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

# Define arguments
parser = argparse.ArgumentParser(description='Script for SVM classification with GloVe and PPMI embeddings')
parser.add_argument('--kernel', type=str, required=True, help='Kernel type for the SVM (e.g., linear, rbf)')
parser.add_argument('--align', type=str, required=True, help='Alignment type (project, cca))')

args = parser.parse_args()

# Define language 
kernel_type = args.kernel
alignment_type = args.align

# Load PPMI
ppmi_type = "all"
ppmi_all_path = "/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn_all/ppmi_embeddings_all.txt"
ppmi = read_embeddings_from_text(ppmi_all_path)

# Iterate through languages
languages = ["sw", "el", "bg", "ur", "th"]
for language in languages:

    # Define embedding paths
    glove_path = f"/ds/text/cc100/vectors_more/vector-{language}.txt"

    # Load dataset (TSV format with columns: gold_label, language, sentence1, sentence2)
    xnli_data = pd.read_csv('./data/xnli.dev.tsv', sep='\t')
    xnli_data_test = pd.read_csv('./data/xnli.test.tsv', sep='\t')
    xnli_data = xnli_data[xnli_data['language'] == language]
    xnli_data_test = xnli_data_test[xnli_data_test['language'] == language]
    print('------------> Data loaded!')

    # Load embeddings
    glove = read_embeddings_from_text(glove_path)
    print('------------> Embeddings loaded!')

    # Align embeddings
    glove_ppmi = svd_alignment(glove, ppmi)

    print(f'------------> Embeddings aligned with {alignment_type} type!')

    xnli_data['sentence1'] = xnli_data['sentence1'].apply(preprocess_text)
    xnli_data_test['sentence1'] = xnli_data_test['sentence1'].apply(preprocess_text)
    xnli_data['sentence2'] = xnli_data['sentence2'].apply(preprocess_text)
    xnli_data_test['sentence2'] = xnli_data_test['sentence2'].apply(preprocess_text)

    # Embed sentence pairs and combine them
    def embed_sentence_pair(row, embedding_model):
        sentence1_embedding = sentence_to_embedding(row['sentence1'], embedding_model)
        sentence2_embedding = sentence_to_embedding(row['sentence2'], embedding_model)
        return np.concatenate((sentence1_embedding, sentence2_embedding))

    # Prepare data
    X_train = xnli_data
    y_train = xnli_data['gold_label']

    X_test = xnli_data_test
    y_test = xnli_data_test['gold_label']

    # Embedding sentences with Glove
    glove_embeddings = xnli_data.apply(lambda row: embed_sentence_pair(row, glove), axis=1).tolist()
    X_train['glove'] = glove_embeddings

    # Embedding sentences with PPMI
    ppmi_embeddings = xnli_data.apply(lambda row: embed_sentence_pair(row, ppmi), axis=1).tolist()
    X_train['ppmi'] = ppmi_embeddings

    # Embedding sentences with GloVe+PPMI
    glove_ppmi_embeddings = xnli_data.apply(lambda row: embed_sentence_pair(row, glove_ppmi), axis=1).tolist()
    X_train['glove_ppmi'] = glove_ppmi_embeddings

    # Do the same for X_test
    glove_embeddings_test = xnli_data_test.apply(lambda row: embed_sentence_pair(row, glove), axis=1).tolist()
    X_test['glove'] = glove_embeddings_test

    ppmi_embeddings_test = xnli_data_test.apply(lambda row: embed_sentence_pair(row, ppmi), axis=1).tolist()
    X_test['ppmi'] = ppmi_embeddings_test

    glove_ppmi_embeddings_test = xnli_data_test.apply(lambda row: embed_sentence_pair(row, glove_ppmi), axis=1).tolist()
    X_test['glove_ppmi'] = glove_ppmi_embeddings_test

    print('------------> Data ready for fitting the models!')

    # Hyperparameter tuning with GridSearchCV
    param_grid = {'C': [1], 'kernel': [kernel_type]}

    # SVM with GLOVE
    print("Tuning SVM with GLOVE...")
    svm_classifier = SVC()
    svm_glove = hyperparam_tuning(np.array(X_train['glove'].tolist()), y_train, svm_classifier, param_grid)
    test_predictions_glove = svm_glove.predict(np.array(X_test['glove'].tolist()))
    print("SVM with GLOVE:")
    print(classification_report(y_test, test_predictions_glove, digits=5))
    save_results(f'./svd_all/{language}/svm_{kernel_type}.txt', y_test, test_predictions_glove)

    # SVM with PPMI
    print("Tuning SVM with PPMI...")
    svm_ppmi = hyperparam_tuning(np.array(X_train['ppmi'].tolist()), y_train, svm_classifier, param_grid)
    test_predictions_ppmi = svm_ppmi.predict(np.array(X_test['ppmi'].tolist()))
    print("SVM with PPMI:")
    print(classification_report(y_test, test_predictions_ppmi, digits=5))
    save_results(f'./svd_all/{language}/svm_{kernel_type}.txt', y_test, test_predictions_ppmi)

    # SVM with GLOVE+PPMI
    print("Tuning SVM with GLOVE+PPMI...")
    svm_glove_ppmi = hyperparam_tuning(np.array(X_train['glove_ppmi'].tolist()), y_train, svm_classifier, param_grid)
    test_predictions_glove_ppmi = svm_glove_ppmi.predict(np.array(X_test['glove_ppmi'].tolist()))
    print("SVM with GLOVE+PPMI:")
    print(classification_report(y_test, test_predictions_glove_ppmi, digits=5))
    save_results(f'./svd_all/{language}/svm_{kernel_type}.txt', y_test, test_predictions_glove_ppmi)