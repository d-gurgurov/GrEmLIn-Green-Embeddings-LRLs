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

# Text preprocessing
import string
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Dataset
from datasets import load_dataset

import fasttext
from nltk import word_tokenize
import numpy as np
import string

# Define arguments
parser = argparse.ArgumentParser(description='Script for SVM classification with GloVe and PPMI embeddings')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')
parser.add_argument('--kernel', type=str, required=True, help='Kernel type for the SVM (e.g., linear, rbf)')

args = parser.parse_args()

# Define language 
language = args.language
kernel_type = args.kernel

# Load dataset
dataset = load_dataset('Davlan/sib200', language)

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
    "quy_Latn": "qu", "lim_Latn": "lim", "wol_Latn": "wo", "gla_Latn": "gd", "mya_Mymr": "my", "ydd_Hebr": "yi",
"hau_Latn": "ha", "snd_Arab": "sd", "som_Latn": "so", "ckb_Arab": "ku", "pbt_Arab": "ps", "khm_Khmr": "km",
"guj_Gujr": "gu", "afr_Latn": "af", "glg_Latn": "gl", "isl_Latn": "is", "kaz_Cyrl": "kk", "azj_Latn": "az", 
"tam_Taml": "ta", "lij_Latn": "lij", "ell_Grek": "el", "ukr_Cyrl": "uk", "srd_Latn": "sc", "grn_Latn": "gn",
"lin_Latn": "lin", "zul_Latn": "zu", "hat_Latn": "ht", "xho_Latn": "xh", "jav_Latn": "jv", "san_Deva": "sa",
"lao_Laoo": "la", "pan_Guru": "pa", "gle_Latn": "ga", "kir_Cyrl": "ky", "epo_Latn": "eo", "kan_Knda": "kn",
"bel_Cyrl": "be", "hye_Armn": "hy", "mal_Mlym": "ml", "est_Latn": "et", "zsm_Latn": "ms", "lit_Latn": "lt",
"tha_Thai": "th"
}

# Load FastText model
ft_model = fasttext.load_model(f'/ds/text/cc100/fasttext/cc.{languages_mapping[language]}.300.bin')

def preprocess_text_fasttext(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    return words

def embed_sentence_fasttext(sentence, model=ft_model):
    """Function that embeds a sentence by summing up the embeddings of each word"""
    words = preprocess_text_fasttext(sentence)
    sentence_embedding = np.zeros(model.get_dimension())  # Initialize a vector of zeros
    for word in words:
        word_embedding = model.get_word_vector(word)
        sentence_embedding += word_embedding  # Sum word embeddings
    return sentence_embedding

# Function to embed all sentences in a dataset
def embed_sentences_fasttext(texts):
    embeddings = []
    for text in texts:
        embedding = embed_sentence_fasttext(text)
        embeddings.append(embedding)
    return np.vstack(embeddings)

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

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categories into numerical labels
train['label'] = label_encoder.fit_transform(train['category'])
valid['label'] = label_encoder.transform(valid['category'])
test['label'] = label_encoder.transform(test['category'])

X_train = train['text']
y_train = train['label']

X_valid = valid['text']
y_valid = valid['label']

X_test = test['text']
y_test = test['label']


# Embed sentences
X_train_embeddings = embed_sentences_fasttext(X_train.tolist())
X_valid_embeddings = embed_sentences_fasttext(X_valid.tolist())
X_test_embeddings = embed_sentences_fasttext(X_test.tolist())

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
print("SVM with FastText embeddings:")
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

save_results(f'./fasttext_rbf_100/{language}/svm_fasttext.txt', y_test, test_predictions)
