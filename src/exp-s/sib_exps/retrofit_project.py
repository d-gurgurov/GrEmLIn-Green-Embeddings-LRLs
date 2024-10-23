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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import MinMaxScaler


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
parser.add_argument('--align', type=str, required=True, help='Alignment type (project, cca))')

args = parser.parse_args()

# Define language 
language = args.language
kernel_type = args.kernel
alignment_type = args.align

# Define language mapping
languages_mapping = {
    "tel_Telu": "te",   # Telugu
    "ben_Beng": "bn",   # Bengali
    "lvs_Latn": "lv",   # Latvian
    "mlt_Latn": "mt",   # Maltese
    "amh_Ethi": "am",   # Amharic
    "uzn_Latn": "uz",   # Uzbek
    "sun_Latn": "su",   # Sundanese
    "cym_Latn": "cy",   # Welsh
    "mar_Deva": "mr",   # Marathi
    "ckb_Arab": "ku",   # Kurdish
    "mkd_Cyrl": "mk",   # Macedonian
    "kat_Geor": "ka",   # Georgian
    "slk_Latn": "sk",   # Slovak
    "ell_Grek": "el",   # Greek
    "tha_Thai": "th",   # Thai
    "azj_Latn": "az",   # Azerbaijani
    "slv_Latn": "sl",   # Slovenian
    "heb_Hebr": "he",   # Hebrew
    "ron_Latn": "ro",   # Romanian
    "dan_Latn": "da",   # Danish
    "urd_Arab": "ur",   # Urdu
    "sin_Sinh": "si",   # Sinhala
    "yor_Latn": "yo",   # Yoruba
    "swh_Latn": "sw",   # Swahili
    "uig_Arab": "ug",   # Uyghur
    "bod_Tibt": "bo",   # Tibetan
    "jav_Latn": "jv",   # Javanese
    "npi_Deva": "ne",   # Nepali
    "bul_Cyrl": "bg",    # Bulgarian
    "quy_Latn": "qu", "lim_Latn": "li", "wol_Latn": "wo", "gla_Latn": "gd", "mya_Mymr": "my", "ydd_Hebr": "yi",
"hau_Latn": "ha", "snd_Arab": "sd", "som_Latn": "so", "ckb_Arab": "ku", "pbt_Arab": "ps", "khm_Khmr": "km",
"guj_Gujr": "gu", "afr_Latn": "af", "glg_Latn": "gl", "isl_Latn": "is", "kaz_Cyrl": "kk", "azj_Latn": "az", 
"tam_Taml": "ta", "lij_Latn": "lv", "ell_Grek": "el", "ukr_Cyrl": "uk", "srd_Latn": "sc", "grn_Latn": "gn",
"lin_Latn": "li", "zul_Latn": "zu", "hat_Latn": "ht", "xho_Latn": "xh", "jav_Latn": "jv", "san_Deva": "sa",
"lao_Laoo": "la", "pan_Guru": "pa", "gle_Latn": "ga", "kir_Cyrl": "ky", "epo_Latn": "eo", "kan_Knda": "kn",
"bel_Cyrl": "be", "hye_Armn": "hy", "mal_Mlym": "ml", "est_Latn": "et", "zsm_Latn": "ms", "lit_Latn": "lt",
"tha_Thai": "th"
}

# Define embedding paths
glove_path = f"/ds/text/cc100/vectors_more/vector-{languages_mapping[language]}.txt"
ppmi_path = f"/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn/ppmi_embeddings_{languages_mapping[language]}.txt"

# Load data
dataset = load_dataset('Davlan/sib200', language)
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
    cca = CCA(n_components=100, scale=True)
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

# Load embeddings
glove = read_embeddings_from_text(glove_path)
ppmi = read_embeddings_from_text(ppmi_path)
print('------------> Embeddings loaded!')

# Align embeddings
if alignment_type == "project":
    glove_ppmi = align_embeddings_transform_matrix(glove, ppmi)

if alignment_type == "cca":
    glove_ppmi, ppmi_glove = cca_alignment(glove, ppmi)

if alignment_type == "svd":
    glove_ppmi = svd_alignment(glove, ppmi)

print(f'------------> Embeddings aligned with {alignment_type} type!')

from sklearn.preprocessing import LabelEncoder

# Preparing data
train = pd.DataFrame(dataset["train"])
valid = pd.DataFrame(dataset["validation"])
test = pd.DataFrame(dataset["test"])

# Check and handle additional labels dynamically
num_labels = len(train['category'].unique())
print(f"Number of unique labels: {num_labels}")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categories into numerical labels
train['label'] = label_encoder.fit_transform(train['category'])
valid['label'] = label_encoder.transform(valid['category'])
test['label'] = label_encoder.transform(test['category'])

train['text'] = train['text'].apply(preprocess_text)
valid['text'] = valid['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

X_train = train.drop(columns=['label']) 
y_train = train['label']

X_valid = valid.drop(columns=['label']) 
y_valid = valid['label']

X_test = test.drop(columns=['label']) 
y_test = test['label']

X_train = pd.DataFrame(X_train, columns=X_train.columns)
y_train = pd.Series(y_train, name='label')

# Embedding sentences with Glove
X_train['glove'] = X_train['text'].apply(lambda x: sentence_to_embedding(x, glove)).tolist()
X_valid['glove'] = X_valid['text'].apply(lambda x: sentence_to_embedding(x, glove)).tolist()
X_test['glove'] = X_test['text'].apply(lambda x: sentence_to_embedding(x, glove)).tolist()

# Embedding sentences with PPMI
X_train['ppmi'] = X_train['text'].apply(lambda x: sentence_to_embedding(x, ppmi)).tolist()
X_valid['ppmi'] = X_valid['text'].apply(lambda x: sentence_to_embedding(x, ppmi)).tolist()
X_test['ppmi'] = X_test['text'].apply(lambda x: sentence_to_embedding(x, ppmi)).tolist()

# Embedding sentences with Glove+PPMI
X_train['glove_ppmi'] = X_train['text'].apply(lambda x: sentence_to_embedding(x, glove_ppmi)).tolist()
X_valid['glove_ppmi'] = X_valid['text'].apply(lambda x: sentence_to_embedding(x, glove_ppmi)).tolist()
X_test['glove_ppmi'] = X_test['text'].apply(lambda x: sentence_to_embedding(x, glove_ppmi)).tolist()

print('------------> Data ready for fitting the models!')

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [1], 'kernel': [kernel_type]}

# SVM with GLOVE
print("Tuning SVM with GLOVE...")
svm_classifier = SVC()
svm_glove = hyperparam_tuning(np.array(X_train['glove'].tolist()), y_train, np.array(X_valid['glove'].tolist()), y_valid, svm_classifier, param_grid)
test_predictions_glove = svm_glove.predict(np.array(X_test['glove'].tolist()))
print("SVM with GLOVE:")
print(classification_report(y_test, test_predictions_glove, digits=3))
save_results(f'./svd_single/svm_{kernel_type}/{language}/svm_{kernel_type}.txt', y_test, test_predictions_glove)

# SVM with PPMI
print("Tuning SVM with PPMI...")
svm_ppmi = hyperparam_tuning(np.array(X_train['ppmi'].tolist()), y_train, np.array(X_valid['ppmi'].tolist()), y_valid, svm_classifier, param_grid)
test_predictions_ppmi = svm_ppmi.predict(np.array(X_test['ppmi'].tolist()))
print("SVM with PPMI:")
print(classification_report(y_test, test_predictions_ppmi, digits=3))
save_results(f'./svd_single/svm_{kernel_type}/{language}/svm_{kernel_type}.txt', y_test, test_predictions_ppmi)

# SVM with GLOVE+PPMI
print("Tuning SVM with GLOVE+PPMI...")
svm_glove_ppmi = hyperparam_tuning(np.array(X_train['glove_ppmi'].tolist()), y_train, np.array(X_valid['glove_ppmi'].tolist()), y_valid, svm_classifier, param_grid)
test_predictions_glove_ppmi = svm_glove_ppmi.predict(np.array(X_test['glove_ppmi'].tolist()))
print("SVM with GLOVE+PPMI:")
print(classification_report(y_test, test_predictions_glove_ppmi, digits=3))
save_results(f'./svd_single/svm_{kernel_type}/{language}/svm_{kernel_type}.txt', y_test, test_predictions_glove_ppmi)