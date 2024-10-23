import json 
import re

def clean_prefixes(data):
    cleaned_data = {}
    prefix_pattern = re.compile(r'^/c/[^/]+/')  # Matches the prefix part '/c/<lang_code>/'

    for lang, words in data.items():
        cleaned_words = {}
        for key, values in words.items():
            # Clean the key
            cleaned_key = prefix_pattern.sub('', key).replace('_', ' ')
                
            # Clean the values
            cleaned_values = [prefix_pattern.sub('', value).replace('_', ' ') for value in values]
                
            cleaned_words[cleaned_key] = cleaned_values
        
        cleaned_data[lang] = cleaned_words
    
    return cleaned_data

# Load the relations dictionary from the JSON file
input_file_path = "cn_relations.json"
with open(input_file_path, "r") as json_file:
    data = json.load(json_file)

cleaned_data = clean_prefixes(data)

print(cleaned_data['zdj'])

# Save the cleaned data to a new JSON file
output_file_path = "cn_relations_clean.json"
with open(output_file_path, "w") as json_file:
    json.dump(cleaned_data, json_file, ensure_ascii=False, indent=4)
