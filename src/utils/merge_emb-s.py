import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datasets import load_dataset

# Define functions
def svd_alignment(embeddings_dict_1, embeddings_dict_2):
    """Function to align embeddings using SVD"""
    common_words = [word for word in embeddings_dict_1 if word in embeddings_dict_2 and len(embeddings_dict_1[word]) == 300]

    print('Common vocabulary between the embedding spaces:', len(common_words))

    # Prepare data for alignment
    embeddings_1 = np.array([embeddings_dict_1[word] for word in common_words])
    minimum = np.min(embeddings_1)
    maximum = np.max(embeddings_1)

    # Normalize embeddings
    embeddings_2 = np.array([embeddings_dict_2[word] for word in common_words])
    embeddings_2 = normalize_embeddings(embeddings_2, minimum, maximum)

    # Concatenate embeddings for SVD
    concatenated_embeddings = np.concatenate((embeddings_1, embeddings_2), axis=1)

    # Perform SVD on concatenated embeddings
    U, S, Vt = np.linalg.svd(concatenated_embeddings, full_matrices=False)
    reduced_embeddings = U[:, :300]

    # Fit linear regression to find transformation matrix
    model = LinearRegression()
    model.fit(embeddings_1, reduced_embeddings)
    transformation = model.coef_.T

    # Transform the first embedding space
    transformed_embeddings = {word: np.dot(embeddings_dict_1[word], transformation) for word in embeddings_dict_1}

    return transformed_embeddings

def normalize_embeddings(embeddings, min_val, max_val):
    """Normalize embeddings to a specified range."""
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    return scaler.fit_transform(embeddings)

def read_embeddings_from_text(file_path, embedding_size=300):
    """Function to read the embeddings from a text file."""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            embedding_start_index = len(parts) - embedding_size
            phrase = ' '.join(parts[:embedding_start_index])
            embedding = np.array([float(val) for val in parts[embedding_start_index:]])
            embeddings[phrase] = embedding
    return embeddings

# Define paths
language = "en"  # Example language
glove_path = f"/path/to/glove/vector-{language}.txt"
ppmi_path = f"/path/to/ppmi_embeddings_{language}.txt"

# Load embeddings
glove = read_embeddings_from_text(glove_path)
ppmi = read_embeddings_from_text(ppmi_path)

print('Embeddings loaded!')

# Align embeddings
alignment_type = "svd"
if alignment_type == "svd":
    aligned_embeddings = svd_alignment(glove, ppmi)

print('Embeddings aligned!')

# Save aligned embeddings
def save_embeddings_to_text(embeddings, file_path):
    """Save embeddings to a text file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for word, vector in embeddings.items():
            vector_str = ' '.join(map(str, vector))
            file.write(f"{word} {vector_str}\n")

output_path = f"/path/to/aligned_embeddings_{language}.txt"
save_embeddings_to_text(aligned_embeddings, output_path)

print(f'Aligned embeddings saved to {output_path}!')
