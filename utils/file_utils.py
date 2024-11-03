import re

def read_data_schema(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_to_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)


def read_data_schema_from_file(file_path):
    """
    Reads the data schema from a file.

    Args:
        file_path (str): The path to the file containing the data schema.

    Returns:
        str: The contents of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    with open(file_path, 'r') as file:
        data_schema = file.read()
    return data_schema

def make_safe_folder_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    # This example replaces anything that is not a letter, number, underscore, or hyphen with an underscore
    safe_folder_name = re.sub(r'[^A-Za-z0-9_\-]', '_', folder_name)
    return safe_folder_name