import json

# Load the relations dictionary from the JSON file
input_file_path = "cn_relations_clean.json"
with open(input_file_path, "r") as json_file:
    relations_dict = json.load(json_file)

# Prepare the results
results = []
num_languages = len(relations_dict)
results.append(f"Number of languages: {num_languages}\n")

for language_code, start_nodes in relations_dict.items():
    num_start_edges = len(start_nodes.keys())
    num_end_edges = len([end_node for edges in start_nodes.values() for end_node in edges])
    results.append(f"Language: {language_code} - Number of start edges: {num_start_edges} - Number of end edges: {num_end_edges}\n")

# Write the results to a text file
output_file_path = "cn_relations_summary.txt"
with open(output_file_path, "w") as txt_file:
    txt_file.writelines(results)

print(f"Summary saved to {output_file_path}")
