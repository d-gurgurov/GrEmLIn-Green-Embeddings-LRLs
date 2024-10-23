import numpy as np
import string
from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from datasets import load_dataset

# Define language 
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

# Load GloVe and PPMI embeddings
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

languages = ["sin_Sinh", "amh_Ethi", "sun_Latn", "swh_Latn", "kat_Geor", "npi_Deva", "uig_Arab", 
"yor_Latn", "urd_Arab", "mkd_Cyrl", "mar_Deva", "ben_Beng", "tel_Telu", "uzn_Latn", "azj_Latn", 
"slk_Latn", "slv_Latn", "ron_Latn", "lvs_Latn", "dan_Latn", "heb_Hebr", "cym_Latn", "bul_Cyrl", 
"quy_Latn", "lim_Latn", "wol_Latn", "gla_Latn", "mya_Mymr", "ydd_Hebr",
"hau_Latn", "snd_Arab", "som_Latn", "ckb_Arab", "pbt_Arab", "khm_Khmr", 
"guj_Gujr", "afr_Latn", "glg_Latn", "isl_Latn", "kaz_Cyrl", "azj_Latn", 
"tam_Taml", "lij_Latn", "ell_Grek", "ukr_Cyrl", "srd_Latn", "grn_Latn", 
"lin_Latn", "zul_Latn", "hat_Latn", "xho_Latn", "jav_Latn", "san_Deva", 
"lao_Laoo", "pan_Guru", "gle_Latn", "kir_Cyrl", "epo_Latn", "kan_Knda", 
"bel_Cyrl", "hye_Armn", "mal_Mlym", "est_Latn", "zsm_Latn", "lit_Latn", "tha_Thai"]


print("Loading PPMI embeddings...")
ppmi_path = "/netscratch/dgurgurov/emnlp2024/multilingual_conceptnet/embeddings/cn_all/ppmi_embeddings_all.txt"
ppmi = read_embeddings_from_text(ppmi_path)
print("PPMI embeddings loaded!")


for language in languages:
    # Define embedding paths
    glove_path = f"/ds/text/cc100/vectors_more/vector-{languages_mapping[language]}.txt"

    # Load SIB200 dataset
    dataset = load_dataset('Davlan/sib200', language)
    print('------------> Data loaded!')

    def preprocess_text(text):
        """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        return ' '.join(words)

    # Preparing data
    train_texts = [preprocess_text(item['text']) for item in dataset['train']]
    test_texts = [preprocess_text(item['text']) for item in dataset['test']]

    # Create vocabulary from train and test texts
    def build_vocabulary(texts):
        vocabulary = set()
        for text in texts:
            words = text.split()
            vocabulary.update(words)
        return vocabulary

    train_vocab = build_vocabulary(train_texts)
    test_vocab = build_vocabulary(test_texts)

    # Combine train and test vocabulary
    combined_vocab = train_vocab.union(test_vocab)

    print("Loading GloVe embeddings...")
    glove = read_embeddings_from_text(glove_path)
    print("GloVe embeddings loaded!")

    # Compute coverage
    total_words = len(combined_vocab)

    glove_coverage = sum(1 for word in combined_vocab if word in glove)
    glove_coverage_ratio = glove_coverage / total_words

    # Compute common vocabulary between GloVe and PPMI
    common_words = [word for word in glove if word in ppmi and len(glove[word]) == 300]

    # Prepare results
    results = {
        "Language": f"{language} ({languages_mapping[language]})",
        "GloVe Coverage Ratio": f"{glove_coverage_ratio:.2%}",
        "Common GloVe-PPMI(All) Vocabulary": len(common_words)
    }

    # Save results to markdown file
    with open(f"./stats/ppmi_all/embedding_coverage_{language}.md", "w") as f:
        f.write("# Focused Embedding Coverage Analysis\n\n")
        f.write("| Language | GloVe Coverage Ratio | Common GloVe-PPMI(All) Vocabulary |\n")
        f.write("|----------|----------------------|------------------------------|\n")
        f.write(f"| {results['Language']} | {results['GloVe Coverage Ratio']} | {results['Common GloVe-PPMI(All) Vocabulary']} |\n")

    print(f"Results saved to embedding_coverage_{language}.md")