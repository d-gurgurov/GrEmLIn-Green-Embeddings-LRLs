<h1 align="center">
  <img style="vertical-align:middle" src="assets/1F9CC_color.png" width="40"/>
  <strong style="color: olive; font-family: 'Courier New', monospace;">GrEmLIn</strong>
</h1>

# GrEmLIn: A Repository of Green Baseline Embeddings for 87 Low-Resource Languages Injected with Multilingual Graph Knowledge

This project augments GloVe embeddings for mid- and low-resource languages with graph knowledge from ConceptNet, as well as provides a centralized repository with pre-trained static "green" word embeddings across diverse languages. These embeddings are available for the languages described in the following table and can be enhanced with graph embeddings stored [here](https://huggingface.co/DGurgurov/conceptnet_embeddings) using the algorithm provided in this repository ([merge_emb-s.py](https://github.com/d-gurgurov/GrEmLIn-Green-Embeddings-LRLs/blob/main/src/utils/merge_emb-s.py)). The embeddings can be accessed on [HuggingFace](https://huggingface.co/DFKI). 

## Language Data Details

| ISO   | Language Name     | Dataset Size | Class |ConceptNet Data|
|-------|-------------------|--------------|-------|---------------|
| ss    | Swati             | 86K          | 1     | ✘             |
| sc    | Sardinian         | 143K         | 1     | ✓             |
| yo    | Yoruba            | 1.1M         | 2     | ✓             |
| gn    | Guarani           | 1.5M         | 1     | ✓             |
| qu    | Quechua           | 1.5M         | 1     | ✓             |
| ns    | Northern Sotho    | 1.8M         | 1     | ✘             |
| li    | Limburgish        | 2.2M         | 1     | ✓             |
| ln    | Lingala           | 2.3M         | 1     | ✓             |
| wo    | Wolof             | 3.6M         | 2     | ✓             |
| zu    | Zulu              | 4.3M         | 2     | ✓             |
| rm    | Romansh           | 4.8M         | 1     | ✓             |
| ig    | Igbo              | 6.6M         | 1     | ✘             |
| lg    | Ganda             | 7.3M         | 1     | ✘             |
| as    | Assamese          | 7.6M         | 1     | ✘             |
| tn    | Tswana            | 8.0M         | 2     | ✘             |
| ht    | Haitian           | 9.1M         | 2     | ✓             |
| om    | Oromo             | 11M          | 1     | ✘             |
| su    | Sundanese         | 15M          | 1     | ✓             |
| bs    | Bosnian           | 18M          | 3     | ✘             |
| br    | Breton            | 21M          | 1     | ✓             |
| gd    | Scottish Gaelic   | 22M          | 1     | ✓             | 
| xh    | Xhosa             | 25M          | 2     | ✓             | 
| mg    | Malagasy          | 29M          | 1     | ✓             | 
| jv    | Javanese          | 37M          | 1     | ✓             | 
| fy    | Frisian           | 38M          | 0     | ✓             | 
| sa    | Sanskrit          | 44M          | 2     | ✓             | 
| my    | Burmese           | 46M          | 1     | ✓             | 
| ug    | Uyghur            | 46M          | 1     | ✓             | 
| yi    | Yiddish           | 51M          | 1     | ✓             | 
| or    | Oriya             | 56M          | 1     | ✓             | 
| ha    | Hausa             | 61M          | 2     | ✓             |  
| la    | Lao               | 63M          | 2     | ✓             | 
| sd    | Sindhi            | 67M          | 1     | ✓             | 
| ta_rom| Tamil Romanized   | 68M          | 3     | ✘             |
| so    | Somali            | 78M          | 1     | ✓             |
| te_rom| Telugu Romanized  | 79M          | 1     | ✘             |
| ku    | Kurdish           | 90M          | 0     | ✓             |
| pu    | Punjabi           | 90M          | 2     | ✓             |
| ps    | Pashto            | 107M         | 1     | ✓             |
| ga    | Irish             | 108M         | 2     | ✓             |
| am    | Amharic           | 133M         | 2     | ✓             |
| ur_rom| Urdu Romanized    | 141M         | 3     | ✘             |
| km    | Khmer             | 153M         | 1     | ✓             |
| uz    | Uzbek             | 155M         | 3     | ✓             |
| bn_rom| Bengali Romanized | 164M         | 3     | ✘             |
| ky    | Kyrgyz            | 173M         | 3     | ✓             |
| my_zaw| Burmese (Zawgyi)  | 178M         | 1     | ✘             |
| cy    | Welsh             | 179M         | 1     | ✓             |
| gu    | Gujarati          | 242M         | 1     | ✓             |
| eo    | Esperanto         | 250M         | 1     | ✓             |
| af    | Afrikaans         | 305M         | 3     | ✓             |
| sw    | Swahili           | 332M         | 2     | ✓             |
| mr    | Marathi           | 334M         | 2     | ✓             |
| kn    | Kannada           | 360M         | 1     | ✓             |
| ne    | Nepali            | 393M         | 1     | ✓             |
| mn    | Mongolian         | 397M         | 1     | ✓             |
| si    | Sinhala           | 452M         | 0     | ✓             |
| te    | Telugu            | 536M         | 1     | ✓             |
| la    | Latin             | 609M         | 3     | ✓             |
| be    | Belarussian       | 692M         | 3     | ✓             |
| tl    | Tagalog           | 701M         | 3     | ✘             |
| mk    | Macedonian        | 706M         | 1     | ✓             |
| gl    | Galician          | 708M         | 3     | ✓             |
| hy    | Armenian          | 776M         | 1     | ✓             |
| is    | Icelandic         | 779M         | 2     | ✓             |
| ml    | Malayalam         | 831M         | 1     | ✓             |
| bn    | Bengali           | 860M         | 3     | ✓             |
| ur    | Urdu              | 884M         | 3     | ✓             |
| kk    | Kazakh            | 889M         | 3     | ✓             |
| ka    | Georgian          | 1.1G         | 3     | ✓             |
| az    | Azerbaijani       | 1.3G         | 1     | ✓             |
| sq    | Albanian          | 1.3G         | 1     | ✓             |
| ta    | Tamil             | 1.3G         | 3     | ✓             |
| et    | Estonian          | 1.7G         | 3     | ✓             |
| lv    | Latvian           | 2.1G         | 3     | ✓             |
| ms    | Malay             | 2.1G         | 3     | ✓             |
| sl    | Slovenian         | 2.8G         | 3     | ✓             |
| lt    | Lithuanian        | 3.4G         | 3     | ✓             |
| he    | Hebrew            | 6.1G         | 3     | ✓             |
| sk    | Slovak            | 6.1G         | 3     | ✓             |
| el    | Greek             | 7.4G         | 3     | ✓             |
| th    | Thai              | 8.7G         | 3     | ✓             |
| bg    | Bulgarian         | 9.3G         | 3     | ✓             |
| da    | Danish            | 12G          | 3     | ✓             |
| uk    | Ukrainian         | 14G          | 3     | ✓             |
| ro    | Romanian          | 16G          | 3     | ✓             |
| id    | Indonesian        | 36G          | 3     | ✘             |


## ConceptNet Data Details

For this project, we extracted all data from the [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads) database. The extraction process involved several steps to clean and analyze the data from the official ConceptNet dump available [here](https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz).

The final extracted dataset is a JSON file representing a dictionary with language codes and start and end edges for each language. Start edges represent the unique words in a target language, while end edges are the words related to the start edges through various types of relationships. The relationship types and sources are not extracted.

The dataset, as well as details on the amount of extracted data for each language, are available on [Hugging Face](https://huggingface.co/datasets/DGurgurov/conceptnet_all).


## Usage

If you use our embedding enhancement method or pre-trained embeddings, please consider citing our preview paper (the full paper is to be published soon):

```bibtex
@misc{gurgurov2024gremlinrepositorygreenbaseline,
      title={GrEmLIn: A Repository of Green Baseline Embeddings for 87 Low-Resource Languages Injected with Multilingual Graph Knowledge}, 
      author={Daniil Gurgurov and Rishu Kumar and Simon Ostermann},
      year={2024},
      eprint={2409.18193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.18193}, 
}
```

The long paper with the extended experimental results will be published soon!

## License

This project is licensed under the Apache License - see the [LICENSE] file for details.
