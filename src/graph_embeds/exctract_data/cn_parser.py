import gzip
import json

# Define the dictionary to store the relations
print("--------> Sequential processing setup!")
relations_dict = {}

# Path to the gzipped file
file_path = "conceptnet-assertions-5.7.0.csv.gz"

# Function to parse each line and extract words and relations
def parse_line(line):
    parts = line.strip().split("\t")
    if len(parts) >= 3:
        # Extract the start and end nodes
        start_node_uri = parts[2]
        end_node_uri = parts[3]
        
        # Extract the language code
        start_lang_code = start_node_uri.split("/")[2]
        language_code = start_lang_code

        # Extract the node names
        start_node = start_node_uri.split("/")[3]
        end_node = end_node_uri.split("/")[3]

        # Filter out end nodes that are http:// links
        if end_node_uri.startswith("http://"):
            return

        # Initialize the language dictionary if it doesn't exist
        if language_code not in relations_dict:
            relations_dict[language_code] = {}

        # Initialize the start node list if it doesn't exist
        if start_node not in relations_dict[language_code]:
            relations_dict[language_code][start_node] = []
        # Add the end node to the start node's list
        relations_dict[language_code][start_node].append(end_node)

# Open the gzipped file and parse each line sequentially
with gzip.open(file_path, "rt") as f:
    print("--------> File opened!")
    for line in f:
        parse_line(line)


# Save the relations dictionary to a JSON file
print("--------> About to save the file!")
output_file_path = "cn_relations.json"
with open(output_file_path, "w") as json_file:
    json.dump(relations_dict, json_file, ensure_ascii=False)

print("Data saved to", output_file_path)