import os

def save_tool_file(location: str, filename: str, code: str) -> str:
    """
    Saves the generated tool code to a file.

    Args:
        location (str): Directory path where the file will be saved.
        filename (str): Name of the file.
        code (str): The Python code content.

    Returns:
        str: The full path of the saved file.
    """
    os.makedirs(location, exist_ok=True)
    file_path = os.path.join(location, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)

    return file_path
