import numpy as np
import pandas as pd
import json
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import lil_matrix


class PPMIGenerator:
    def __init__(self, language, ppmi_dim):
        """
        Initialize the PPMIGenerator object with a language code and PPMI dimension.

        Args:
        - language_code: A string representing the language code.
        - ppmi_dim: An integer representing the desired dimension for PPMI embeddings.
        """
        self.language = language
        self.ppmi_dim = ppmi_dim

    
    def read_from_json(self, path):
        """Read data from a JSON file."""
        with open(f'{path}', 'r') as f:
            data = json.load(f)
        return data


    def process_conceptnet_data(self, data):
        """
        Process ConceptNet data to extract relationships and prepare for PPMI computation.

        Args:
        - data: A dictionary containing ConceptNet data.

        Returns:
        A dictionary prepared for PPMI computation.
        """
        conceptnet_data = {}

        for lang_data in data.values():
            for start_word, end_words in lang_data.items():
                related_words = set(end_words)
                conceptnet_data[start_word] = list(related_words)

        data_ppmi = {}
        for key, values in conceptnet_data.items():
            new_values = []
            for value in values:
                words = value.split()
                new_values.extend(words)
            data_ppmi[key] = new_values

        return data_ppmi
    
    def process_conceptnet_language_specific(self, conceptnet_data, language):
        """
        Process ConceptNet data to extract relationships for a specific language and prepare for PPMI computation.

        Args:
        - data: A dictionary containing ConceptNet data.

        Returns:
        A dictionary prepared for PPMI computation.
        """

        data_ppmi = {}
        for key, values in conceptnet_data[language].items():
            new_values = []
            for value in values:
                words = value.split()
                new_values.extend(words)
            data_ppmi[key] = new_values

        data_ppmi

        return data_ppmi
    

    def build_ppmi(self, data):
        """
        Build Positive Pointwise Mutual Information (PPMI) embeddings.

        Args:
        - data: A dictionary containing prepared ConceptNet data.

        Returns:
        A dictionary with PPMI embeddings.
        """
        sparse_csr, index = self._build_from_conceptnet_data(data)
        ppmi = self._counts_to_ppmi(sparse_csr)
        u, s, vT = linalg.svds(ppmi, self.ppmi_dim)
        v = vT.T
        values = (u + v) * (s ** 0.5)

        output = pd.DataFrame(values, index=index)

        words_list = list(data.keys())
        ppmi_filtered = output.loc[words_list]
        ppmi_embeddings_dict = ppmi_filtered.apply(lambda x: x.tolist(), axis=1).to_dict()

        return ppmi_embeddings_dict
    

    def _build_from_conceptnet_data(self, data):
        """
        Build a co-occurrence matrix from ConceptNet data. Convert the co-occurrence matrix 
        to a sparse CSR matrix for more efficient computations.

        Args:
        - data: A dictionary containing ConceptNet data.
                                            
        Returns:
        A sparse CSR matrix and a list of all unique words.
        """
        all_words = set(data.keys())
        for related_words in data.values():
            all_words.update(related_words)
        all_words = list(all_words)
        word_to_index = {word: i for i, word in enumerate(all_words)}

        # cooccurrence_matrix = np.zeros((len(all_words), len(all_words)), dtype='float32')
        # lil_matrix works better for big matrices
        cooccurrence_matrix = lil_matrix((len(all_words), len(all_words)), dtype='float32')
        for word, related_words in data.items():
            for related_word in related_words:
                i = word_to_index[word]
                j = word_to_index[related_word]
                cooccurrence_matrix[i, j] += 1

        sparse_csr = sparse.csr_matrix(cooccurrence_matrix)
        return sparse_csr, all_words

    def _counts_to_ppmi(self, counts_csr, smoothing=0.75):
        """
        Convert counts matrix to Positive Pointwise Mutual Information (PPMI) matrix.

        Args:
        - counts_csr: Sparse co-occurrence matrix.
        - smoothing: Smoothing factor for PPMI computation.

        Returns:
        A PPMI matrix.
        """
        # word_counts adds up the total amount of association for each term
        word_counts = np.asarray(counts_csr.sum(axis=1)).flatten()

        # smooth_context_freqs represents the relative frequency of occurrence
        # of each term as a context (a column of the table)
        smooth_context_freqs = np.asarray(counts_csr.sum(axis=0)).flatten() ** smoothing
        smooth_context_freqs /= smooth_context_freqs.sum()

        # divide each row of counts_csr by the word counts. we accomplish this by
        # multiplying on the left by the sparse diagonal matrix of 1 / word_counts.
        smoothing_constant = 1e-10 
        ppmi = sparse.diags(1 / (word_counts + smoothing_constant)).dot(counts_csr)

        # then, similarly divide the columns by smooth_context_freqs, by the same
        # method except that we multiply on the right.
        ppmi = ppmi.dot(sparse.diags(1 / (smooth_context_freqs + smoothing_constant)))

        # take the log of the resulting entries to give pointwise mutual
        # information. Discard those whose PMI is less than 0, to give positive
        # pointwise mutual information (PPMI).
        ppmi.data = np.maximum(np.log(ppmi.data), 0)
        ppmi.eliminate_zeros()
        return ppmi
    

    def save_to_txt(self, data):
        """
        Save PPMI embeddings to a text file.

        Args:
        - data: A dictionary containing PPMI embeddings.
        """
        file_path = f"ppmi_embeddings_{self.language}.txt"

        with open(file_path, 'w') as txt_file:
            ppmi_embeddings_dict = self.build_ppmi(data)
            for key, value in ppmi_embeddings_dict.items():
                values_str = ' '.join(str(val) for val in value)
                txt_file.write(f'{key} {values_str}\n')

        print("PPMI embeddings saved to", file_path)
