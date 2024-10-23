import pandas as pd
import fasttext
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load MultiSimLex data
def load_multisimlex(lang="CYM"):
    # Load MultiSimLex dataset
    simlex = pd.read_csv("./data/scores.csv", delimiter=",")
    simlex_translation = pd.read_csv("./data/translation.csv", delimiter=",")
    
    words_1 = simlex_translation[f"{lang} 1"].tolist()
    words_2 = simlex_translation[f"{lang} 2"].tolist()
    similarity_scores = simlex[f"{lang}"].tolist()
    
    return words_1, words_2, similarity_scores

# Load FastText model
def load_fasttext_model(language):
    model = fasttext.load_model(f'/ds/text/cc100/fasttext/cc.{language}.300.bin')
    return model

# Load GloVe
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

def normalize_embeddings(embeddings, min, max):
    """Normalize embeddings to be in the range from -1 to 1."""
    scaler = MinMaxScaler(feature_range=(min, max))
    return scaler.fit_transform(embeddings)

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

# Get word vector
def get_word_vector_fasttext(model, word):
    try:
        return model.get_word_vector(word)
    except Exception as e:
        print(f"Error retrieving vector for {word}: {e}")
        return np.zeros(model.get_dimension())  

# Compute similarity
def cosine_similarity(vec1, vec2):
    if np.any(vec1) and np.any(vec2):  # Check if vectors are not all zeros
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 0.0

# Evaluate embeddings
def evaluate_embeddings(words_1, words_2, similarity_scores, model, embedding_type="fasttext"):
    predicted_similarities = []
    missing_words = 0
    
    for word1, word2 in zip(words_1, words_2):
        if embedding_type == "fasttext":
            vec1 = get_word_vector_fasttext(model, word1)
            vec2 = get_word_vector_fasttext(model, word2)

        if embedding_type == "glove":
            vec1 = model.get(word1, np.zeros(300))
            vec2 = model.get(word2, np.zeros(300))
        
        similarity = cosine_similarity(vec1, vec2)
        if similarity == 0.0:
            missing_words += 1
        predicted_similarities.append(similarity)
    
    # Spearman correlation
    spearman_corr = spearmanr(similarity_scores, predicted_similarities).correlation
    return spearman_corr, missing_words

ppmi_model_all = read_embeddings_from_text(f"/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn_all/ppmi_embeddings_all.txt")

# Main function
if __name__ == "__main__":
    # Choose languages for evaluation
    languages = ['et', 'cy', 'sw', 'he']
    languages_simlex = ['EST', 'CYM', 'SWA', 'HEB']

    # Iterate through both lists simultaneously
    for language, language_simlex in zip(languages, languages_simlex):
        print(f"Evaluating for language: {language} ({language_simlex})")

        # Load MultiSimLex data
        words_1, words_2, similarity_scores = load_multisimlex(lang=language_simlex)

        # Load FastText model
        ft_model = load_fasttext_model(language)

        glove_model = read_embeddings_from_text(f"/ds/text/cc100/vectors_more/vector-{language}.txt")

        ppmi_model_single = read_embeddings_from_text(f"/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn/ppmi_embeddings_{language}.txt")

        glove_ppmi_single = svd_alignment(glove_model, ppmi_model_single)

        glove_ppmi_all = svd_alignment(glove_model, ppmi_model_all)

        # Evaluate FastText embeddings
        spearman_fasttext, missing_ft = evaluate_embeddings(words_1, words_2, similarity_scores, ft_model, embedding_type="fasttext")
        print(f"FastText Spearman Correlation: {spearman_fasttext:.4f}, Missing words: {missing_ft}")

        # Evaluate GloVe embeddings
        spearman_glove, missing_glove = evaluate_embeddings(words_1, words_2, similarity_scores, glove_model, embedding_type="glove")
        print(f"GloVe Spearman Correlation: {spearman_glove:.4f}, Missing words: {missing_glove}")

        # Evaluate GloVe+PPMI(S) embeddings
        spearman_glove_ppmi_s, missing_glove_ppmi_s = evaluate_embeddings(words_1, words_2, similarity_scores, glove_ppmi_single, embedding_type="glove")
        print(f"GloVe+PPMI(S) Spearman Correlation: {spearman_glove_ppmi_s:.4f}, Missing words: {missing_glove_ppmi_s}")

        # Evaluate GloVe+PPMI(All) embeddings
        spearman_glove_ppmi_all, missing_glove_ppmi_all = evaluate_embeddings(words_1, words_2, similarity_scores, glove_ppmi_all, embedding_type="glove")
        print(f"GloVe+PPMI(All) Spearman Correlation: {spearman_glove_ppmi_all:.4f}, Missing words: {missing_glove_ppmi_all}")
