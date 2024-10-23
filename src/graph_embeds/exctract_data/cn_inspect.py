import json
from itertools import chain

# Load the relations dictionary from the JSON file
input_file_path = "cn_relations.json"
with open(input_file_path, "r") as json_file:
    relations_dict = json.load(json_file)

# Define the specific language to look for
specific_language = "ii"

# Check if the specific language exists in the dictionary
if specific_language in relations_dict:
    start_nodes = relations_dict[specific_language]
    print(start_nodes)
    num_start_edges = sum(len(edges) for edges in start_nodes.values())
    all_end_edges = list(chain.from_iterable(start_nodes.values()))
    num_start_edges = len(start_nodes.keys())
    num_end_edges = len([end_node for edges in start_nodes.values() for end_node in edges])
    print(f"Language: {specific_language} - Number of start edges: {num_start_edges} - Number of end edges: {num_end_edges}\n")


else:
    print(f"Language {specific_language} not found in the dataset.")
