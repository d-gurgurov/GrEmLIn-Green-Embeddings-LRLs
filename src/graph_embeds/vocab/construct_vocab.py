def save_vocab_to_file(vocab, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in vocab:
            file.write(word + '\n')

def read_embeddings_from_text(file_path):
    """
        Read embeddings from a text file.

        Args:
        - file_path: Path to the text file containing embeddings.

        Returns:
        A dictionary of word embeddings.
    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(' ')
            word = parts[0]
            embedding = " "
            embeddings[word] = embedding
    return embeddings

# Specify the path to the embed file
file_path = 'ppmi_embeddings_all.txt'

# Specify the output file path for the vocabulary
output_file = 'vocab.txt'

# Calculate the vocabulary and vocabulary size
embs = read_embeddings_from_text(file_path)
save_vocab_to_file(embs.keys(), output_file=output_file)

# Print the vocabulary size and the path where the vocabulary is saved
print("Vocabulary Size:", len(embs.keys())) # 7381409
print("Vocabulary saved to:", output_file)
