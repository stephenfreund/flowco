import io
from typing import Dict
import zipfile

import os

import requests
from flowco.util.output import error, log, logger

from flowco.session.session_file_system import (
    fs_exists,
    fs_glob,
    fs_ls,
    fs_read,
    fs_write,
)


def get_flowco_files():
    return fs_glob("", "*.flowco")


# def make_default_files():
#     path = os.path.join(os.path.dirname(__file__), "initial_files")
#     for file in os.listdir(path):
#         print(f"Making {file}")
#         file_path = os.path.join(path, file)
#         with open(file_path, "r") as f:
#             content = f.read()
#             fs_write(file, content)


def make_default_files():
    try:
        # Will fail over if the environment variable is not set.
        folder_id = os.environ["GOOGLE_DRIVE_FOLDER_ID"]
        api_key = os.environ["GOOGLE_API_KEY"]

        # Endpoint for listing files in a folder.
        list_url = f"https://www.googleapis.com/drive/v3/files/{folder_id}"
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "key": api_key,
            "fields": "files(id, name, mimeType)",  # adjust fields as needed
        }

        response = requests.get(list_url, params=params)
        response.raise_for_status()  # Raise an error if the request failed
        data = response.json()

        with logger("Making default files from Google Drive"):
            for file in data.get("files", []):
                log(f"Making {file['name']}")
                file_id = file["id"]
                # Build the URL to download file content.
                download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                download_params = {"alt": "media", "key": api_key}
                file_response = requests.get(download_url, params=download_params)
                file_response.raise_for_status()

                # Assuming the file is text-based.
                content = file_response.text
                fs_write(file["name"], content)
    except Exception as e:
        dir = os.path.join(os.path.dirname(__file__), "initial_files")
        # Fallback to local initial files if Google Drive fails
        with logger("Making default files from package"):
            for file in os.listdir(dir):
                log(f"Making {file} from local initial files")
                file_path = os.path.join(dir, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    fs_write(file, content)


def setup_flowco_files() -> bool:
    if not fs_exists("welcome.flowco"):
        make_default_files()
        return True
    else:
        return False


def create_zip_in_memory(files, additional_entries: Dict[str, str] = {}):
    # Create an in-memory bytes buffer
    zip_buffer = io.BytesIO()

    # Create a ZipFile object using the buffer
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            # Add file to the ZIP archive
            zipf.writestr(file, fs_read(file, use_cache=True))
        for file, content in additional_entries.items():
            zipf.writestr(file, content)

    # Retrieve the bytes of the ZIP archive
    zip_data = zip_buffer.getvalue()
    zip_buffer.close()

    return zip_data
