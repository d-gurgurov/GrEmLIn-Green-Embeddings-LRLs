import os
import glob
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Initialize API and repository
login(token='*')

api = HfApi()

def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def get_file_size_gb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Convert bytes to GB

def upload_embeddings(directory):
    # Get all .txt and .bin files in the directory
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    bin_files = glob.glob(os.path.join(directory, "*.bin"))
    
    for txt_file in tqdm(txt_files, desc="Uploading embeddings"):
        # Extract language code from filename
        lang_code = os.path.splitext(os.path.basename(txt_file))[0].split('-')[-1]
        
        bin_file = next((f for f in bin_files if f.endswith(f"-{lang_code}.bin")), None)
        
        if not bin_file:
            print(f"Warning: No corresponding .bin file found for {txt_file}")
            continue
        
        # Check file sizes
        txt_size = get_file_size_gb(txt_file)
        bin_size = get_file_size_gb(bin_file)
        
        if txt_size > 50 or bin_size > 50:
            print(f"Skipping {lang_code} embeddings: File size exceeds 50 GB")
            continue
        
        # Calculate vocabulary size
        vocab_size = count_lines(txt_file)
        
        # Create repository name
        repo_name = f"glove-{lang_code}-cc100"
        
        # Create repository
        api.create_repo(repo_id=repo_name, exist_ok=True)

        
        # Create README content
        readme_content = f"""
---
tags:
  - embeddings
  - glove
  - cc100
license: cc-by-sa-4.0
---
        
# CC100 GloVe Embeddings for {lang_code.upper()} Language

## Model Description
- **Language:** {lang_code}
- **Embedding Algorithm:** GloVe (Global Vectors for Word Representation)
- **Vocabulary Size:** {vocab_size}
- **Vector Dimensions:** 300
- **Training Data:** CC100 dataset

## Training Information
We trained GloVe embeddings using the original C code. The model was trained by stochastically sampling nonzero elements from the co-occurrence matrix, over 100 iterations, to produce 300-dimensional vectors. We used a context window of ten words to the left and ten words to the right. Words with fewer than 5 co-occurrences were excluded for languages with over 1 million tokens in the training data, and the threshold was set to 2 for languages with smaller datasets.

We used data from CC100 for training the static word embeddings. We set xmax = 100, α = 3/4, and used AdaGrad optimization with an initial learning rate of 0.05.

## Usage
These embeddings can be used for various NLP tasks such as text classification, named entity recognition, and as input features for neural networks.

## Citation
If you use these embeddings in your research, please cite:

```bibtex
@misc{{gurgurov2024lowremrepositorywordembeddings,
      title={{LowREm: A Repository of Word Embeddings for 87 Low-Resource Languages Enhanced with Multilingual Graph Knowledge}}, 
      author={{Daniil Gurgurov and Rishu Kumar and Simon Ostermann}},
      year={{2024}},
      eprint={{2409.18193}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}},
      url={{https://arxiv.org/abs/2409.18193}}, 
}}
```

## License
These embeddings are released under the [CC-BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
"""
        
        # Upload files
        api.upload_file(
            path_or_fileobj=txt_file,
            path_in_repo=f"{lang_code}_embeddings.txt",
            repo_id=f"DGurgurov/{repo_name}"
        )
        
        api.upload_file(
            path_or_fileobj=bin_file,
            path_in_repo=f"{lang_code}_embeddings.bin",
            repo_id=f"DGurgurov/{repo_name}"
        )
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=f"DGurgurov/{repo_name}"
        )
        
        print(f"Uploaded embeddings for {lang_code}")

if __name__ == "__main__":
    vectors_more_dir = "/ds/text/cc100/vectors_more"
    
    print("Uploading embeddings from vectors_more directory...")
    upload_embeddings(vectors_more_dir)

print("All embeddings have been uploaded to Hugging Face.")