#!/bin/bash

pip install nltk fasttext

# List of languages
languages=("sin_Sinh" "amh_Ethi" "sun_Latn" "swh_Latn" "kat_Geor" "npi_Deva" "uig_Arab" 
"yor_Latn" "urd_Arab" "mkd_Cyrl" "mar_Deva" "ben_Beng" "tel_Telu" "uzn_Latn" "azj_Latn" 
"slk_Latn" "slv_Latn" "ron_Latn" "lvs_Latn" "dan_Latn" "heb_Hebr" "cym_Latn" "bul_Cyrl" 
"quy_Latn" "lim_Latn" "wol_Latn" "gla_Latn" "mya_Mymr" "ydd_Hebr" 
"hau_Latn" "snd_Arab" "som_Latn" "ckb_Arab" "pbt_Arab" "khm_Khmr" 
"guj_Gujr" "afr_Latn" "glg_Latn" "isl_Latn" "kaz_Cyrl" "azj_Latn" 
"tam_Taml" "lij_Latn" "ell_Grek" "ukr_Cyrl" "srd_Latn" "grn_Latn" 
"lin_Latn" "zul_Latn" "hat_Latn" "xho_Latn" "jav_Latn" "san_Deva" 
"lao_Laoo" "pan_Guru" "gle_Latn" "kir_Cyrl" "epo_Latn" "kan_Knda" 
"bel_Cyrl" "hye_Armn" "mal_Mlym" "est_Latn" "zsm_Latn" "lit_Latn" "tha_Thai")

# Iterate through each language and run the Python script
for language in "${languages[@]}"; do
    echo "Running embs_stats.py for language: $language"
    python embs_stats.py --language "$language" --ppmi_type "all"
done