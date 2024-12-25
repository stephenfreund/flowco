import io
import zipfile

import os
from flowco.session.session_file_system import (
    fs_exists,
    fs_glob,
    fs_ls,
    fs_read,
    fs_write,
)


def get_flowco_files():
    return fs_glob("", "*.flowco")


def make_default_files():
    path = os.path.join(os.path.dirname(__file__), "initial_files")
    for file in os.listdir(path):
        print(f"Making {file}")
        file_path = os.path.join(path, file)
        with open(file_path, "r") as f:
            content = f.read()
            fs_write(file, content)


def setup_flowco_files():
    if not fs_exists("welcome.flowco"):
        make_default_files()


def create_zip_in_memory(files):
    """
    Create a ZIP archive containing specified files, and return it as bytes.

    Parameters:
    - files: List of file paths to include in the ZIP file.

    Returns:
    - zip_data: Bytes object containing the ZIP archive.
    """
    # Create an in-memory bytes buffer
    zip_buffer = io.BytesIO()

    # Create a ZipFile object using the buffer
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            # Add file to the ZIP archive
            zipf.writestr(file, fs_read(file, use_cache=True))

    # Retrieve the bytes of the ZIP archive
    zip_data = zip_buffer.getvalue()
    zip_buffer.close()

    return zip_data
