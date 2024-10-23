import re

def read_and_format_table(file_path):
    # Read the file contents
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize the list to store table rows
    table_rows = []

    # Regex pattern to extract the required information
    pattern = re.compile(r"Language: (\w{2,3}) - Number of start edges: (\d+) - Number of end edges: (\d+)")

    # Extract information from each line
    for line in lines:
        match = pattern.search(line)
        if match:
            language_code = match.group(1)
            start_edges = match.group(2)
            end_edges = match.group(3)
            table_rows.append([language_code, start_edges, end_edges])

    # Determine the maximum width of each column
    col_widths = [max(len(row[i]) for row in table_rows) for i in range(3)]

    # Print the header row
    header = ["Language Code", "Start Edges", "End Edges"]
    header_row = "| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(3)) + " |"
    separator_row = "| " + " | ".join('-' * col_widths[i] for i in range(3)) + " |"

    print(header_row)
    print(separator_row)

    # Print the table rows
    for row in table_rows:
        formatted_row = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(3)) + " |"
        print(formatted_row)

# Specify the path to your input file
file_path = 'cn_relations_summary.txt'
read_and_format_table(file_path)
