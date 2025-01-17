import os
import shutil
import stat
from tempfile import TemporaryDirectory
from typing import List
from flowco.util.output import log


class Sandbox:
    def __init__(self, files: List[str] | None = None):
        if files is None:
            files = []
        self.sandbox_dir = TemporaryDirectory()
        self.files = files
        self.restore()

    def get_sandbox_path(self):
        return self.sandbox_dir.name

    def temporary_file(self, filename: str):
        return os.path.join(self.sandbox_dir.name, filename)

    def __enter__(self):
        log(f"Entering sandbox {self.sandbox_dir.name}")
        self.sandbox_dir.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        log(f"Exiting sandbox {self.sandbox_dir.name}")
        return self.sandbox_dir.__exit__(exc_type, exc_value, traceback)

    def cleanup(self):
        self.sandbox_dir.cleanup()

    def restore(self):
        # Remove all files and directories in the sandbox
        for item in os.listdir(self.sandbox_dir.name):
            item_path = os.path.join(self.sandbox_dir.name, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        # Copy all files and directories from self.files to the sandbox
        for file in self.files:
            log(f"Copying '{file}' to sandbox")
            basename = os.path.basename(file)
            dest_path = os.path.join(self.sandbox_dir.name, basename)

            if os.path.isdir(file):
                shutil.copytree(file, dest_path)
                self._make_directory_read_only(dest_path)
            else:
                shutil.copy2(file, dest_path)  # Use copy2 to preserve metadata
                self._make_file_read_only(dest_path)

    def _make_file_read_only(self, file_path):
        """
        Change the file permissions to read-only.
        """
        os.chmod(file_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    def _make_directory_read_only(self, dir_path):
        """
        Recursively change directory and all contained files to read-only.
        """
        for root, dirs, files in os.walk(dir_path):
            for dirname in dirs:
                dir_full_path = os.path.join(root, dirname)
                os.chmod(dir_full_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            for filename in files:
                file_full_path = os.path.join(root, filename)
                os.chmod(file_full_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        # Optionally, set the top directory to read-only as well
        os.chmod(dir_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


# Example usage:
if __name__ == "__main__":
    source_files = ["source.txt", "another_file.py", "some_directory"]
    with Sandbox(files=source_files) as sandbox:
        # Perform operations within the sandbox
        sandbox_path = sandbox.sandbox_dir.name
        print(f"Sandbox is at: {sandbox_path}")
        # You can verify that the files are read-only here
