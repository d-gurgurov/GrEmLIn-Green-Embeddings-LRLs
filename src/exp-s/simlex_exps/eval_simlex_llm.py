from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import torch

# Load MultiSimLex data
def load_multisimlex(lang="CYM"):
    # Load MultiSimLex dataset
    simlex = pd.read_csv("./data/scores.csv", delimiter=",")
    simlex_translation = pd.read_csv("./data/translation.csv", delimiter=",")
    
    words_1 = simlex_translation[f"{lang} 1"].tolist()
    words_2 = simlex_translation[f"{lang} 2"].tolist()
    similarity_scores = simlex[f"{lang}"].tolist()
    
    return words_1, words_2, similarity_scores

# Load XLM-R model and tokenizer
xlmr_model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-base').to('cuda')
xlmr_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

# Load E5 model
e5_model = SentenceTransformer('intfloat/multilingual-e5-base')

# Compute similarity
def cosine_similarity(vec1, vec2):
    if np.any(vec1) and np.any(vec2):  # Check if vectors are not all zeros
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 0.0

def get_xlmr_embedding(text):
    if not isinstance(text, str):
        print(f"Invalid input for XLM-R embedding: {text}")
        return np.zeros(xlmr_model.config.hidden_size)  # Return a zero vector of appropriate size
    inputs = xlmr_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = xlmr_model(**inputs)
    return outputs.last_hidden_state.sum(dim=1).cpu().numpy().flatten()

def get_e5_embedding(text):
    if not isinstance(text, str):
        print(f"Invalid input for E5 embedding: {text}")
        return np.zeros(e5_model.get_sentence_embedding_dimension())  # Return a zero vector of appropriate size
    return e5_model.encode(text, convert_to_tensor=True).cpu().numpy().flatten()

def evaluate_embeddings(words_1, words_2, similarity_scores, embedding_type="fasttext"):
    predicted_similarities = []
    missing_words = 0
    
    for word1, word2 in zip(words_1, words_2):
        if embedding_type == "xlmr":
            vec1 = get_xlmr_embedding(word1)
            vec2 = get_xlmr_embedding(word2)
        elif embedding_type == "e5":
            vec1 = get_e5_embedding(word1)
            vec2 = get_e5_embedding(word2)

        similarity = cosine_similarity(vec1, vec2)
        if similarity == 0.0:
            missing_words += 1
        predicted_similarities.append(similarity)
    
    # Spearman correlation
    spearman_corr = spearmanr(similarity_scores, predicted_similarities).correlation
    return spearman_corr, missing_words


if __name__ == "__main__":
    # Choose languages for evaluation
    languages = ['et', 'cy', 'sw', 'he']
    languages_simlex = ['EST', 'CYM', 'SWA', 'HEB']

    # Iterate through both lists simultaneously
    for language, language_simlex in zip(languages, languages_simlex):
        print(f"Evaluating for language: {language} ({language_simlex})")

        # Load MultiSimLex data
        words_1, words_2, similarity_scores = load_multisimlex(lang=language_simlex)

        # Evaluate XLM-R embeddings
        spearman_xlmr, missing_xlmr = evaluate_embeddings(words_1, words_2, similarity_scores, embedding_type="xlmr")
        print(f"XLM-R Spearman Correlation: {spearman_xlmr:.4f}, Missing words: {missing_xlmr}")

        # Evaluate E5 embeddings
        spearman_e5, missing_e5 = evaluate_embeddings(words_1, words_2, similarity_scores, embedding_type="e5")
        print(f"E5 Spearman Correlation: {spearman_e5:.4f}, Missing words: {missing_e5}")