import uuid
import fsspec
import posixpath
from typing import Any, List
import threading
import hashlib

from flowco.session.session import session


class SessionFileSystem:
    """
    A filesystem abstraction using fsspec that supports local and S3 filesystems.

    Attributes:
        root (str): The base path to all files. Can be a local path (e.g., 'file:///path')
                    or an S3 path (e.g., 's3://bucket/path').
    """

    @staticmethod
    def make_unique_path(fs_root: str, base: str) -> str:
        """
        Generates a unique path by appending a UUID to the given path.

        Args:
            base (str): The base path to which a UUID will be appended.

        Returns:
            str: A unique path.
        """
        fs, root_path = fsspec.core.url_to_fs(fs_root)
        # Ensure the root_path ends with a '/'
        if not root_path.endswith("/"):
            root_path += "/"

        print(root_path)

        while True:
            path = f"{root_path}{base}{uuid.uuid4().hex[:8]}/"
            try:
                fs.makedirs(path, exist_ok=False)
                return path
            except FileExistsError:
                print(f"Directory {path} already exists, generating a new one.")
                # If the directory already exists, generate a new one
                pass

    def __init__(self, root: str):
        """
        Initializes the SessionFileSystem object with the specified root path.

        Args:
            root (str): The base path for all file operations.
        """
        self.root = root
        # Initialize the filesystem and extract the root path
        self.fs, self.root_path = fsspec.core.url_to_fs(self.root)

        # Ensure the root_path ends with a '/'
        if not self.root_path.endswith("/"):
            self.root_path += "/"

        # Initialize the cache dictionary
        # Structure: { rel_path: { 'mtime': modification_time, 'data': file_content } }
        self.cache = {}

        # A lock to make cache access thread-safe
        self.lock = threading.Lock()

        if not self.fs.exists(self.root_path):
            self.fs.makedirs(self.root_path)

    def _full_path(self, rel_path: str) -> str:
        """
        Constructs the full path by joining the root path with the relative path.

        Args:
            rel_path (str): The relative path to append to the root.

        Returns:
            str: The combined full path.
        """
        return posixpath.join(self.root_path, rel_path)

    def read(self, rel_path: str, mode: str = "r", use_cache: bool = False) -> Any:
        """
        Reads the content of the file at the specified relative path.

        Args:
            rel_path (str): The relative path to the file.
            mode (str, optional): The mode in which to open the file. Defaults to 'r'.
            use_cache (bool, optional): Whether to use the cached version if available and valid.
                                        Defaults to False.

        Returns:
            Any: The content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file's modification time is unavailable when using cache.
        """
        full_path = self._full_path(rel_path)

        if use_cache:
            with self.lock:
                try:
                    # Fetch the file's metadata to get the last modification time
                    mtime = self._get_file_modification_time(full_path)
                    if mtime is None:
                        raise ValueError(
                            f"Modification time not available for {full_path}"
                        )
                except FileNotFoundError:
                    # If the file doesn't exist, remove it from cache if present
                    self.cache.pop(rel_path, None)
                    raise

                cached = self.cache.get(rel_path)
                if cached and cached["mtime"] == mtime:
                    # Return cached data if the file hasn't changed
                    # print("cache hit")
                    return cached["data"]
                else:
                    # Read the file and update the cache
                    with self.fs.open(full_path, mode) as f:
                        data = f.read()
                    self.cache[rel_path] = {"mtime": mtime, "data": data}
                    return data
        else:
            # Directly read the file without using cache
            with self.fs.open(full_path, mode) as f:
                return f.read()

    def write(self, rel_path: str, data: Any, mode: str = "w") -> None:
        """
        Writes data to the file at the specified relative path.

        Args:
            rel_path (str): The relative path to the file.
            data (Any): The data to write to the file.
            mode (str, optional): The mode in which to open the file. Defaults to 'w'.
        """
        full_path = self._full_path(rel_path)
        with self.fs.open(full_path, mode) as f:
            f.write(data)

        # Update the cache with the new data and modification time
        with self.lock:
            try:
                mtime = self._get_file_modification_time(full_path)
                if mtime is not None:
                    self.cache[rel_path] = {"mtime": mtime, "data": data}
                else:
                    # If modification time is unavailable, remove from cache
                    self.cache.pop(rel_path, None)
            except FileNotFoundError:
                # If the file was deleted after writing, remove from cache
                self.cache.pop(rel_path, None)

    def _get_file_modification_time(self, full_path):
        info = self.fs.info(full_path)
        mtime = info.get("mtime", None)
        if mtime is None:
            mtime = info.get("LastModified", None)
            if mtime is None:
                raise FileNotFoundError(
                    f"Modification time not available for {full_path}"
                )
        return mtime

    def ls(self, rel_path: str = "") -> List[str]:
        """
        Lists all files and directories in the specified directory relative to the root path.

        Args:
            rel_path (str, optional): The relative directory path to list files from.
                                       Defaults to the root directory.

        Returns:
            List[str]: A list of relative file and directory paths.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        full_path = self._full_path(rel_path)
        try:
            # List all files and directories
            files = self.fs.ls(full_path, detail=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory '{rel_path}' does not exist.")

        # Convert full paths back to relative paths
        relative_files = [posixpath.relpath(file, self.root_path) for file in files]
        return relative_files

    def exists(self, rel_path: str) -> bool:
        """
        Checks if a file or directory exists at the specified relative path.

        Args:
            rel_path (str): The relative path to the file or directory.

        Returns:
            bool: True if the file or directory exists, False otherwise.
        """
        full_path = self._full_path(rel_path)
        return self.fs.exists(full_path)

    def glob(self, rel_path: str, pattern: str) -> List[str]:
        """
        Lists files matching a specific glob pattern within the specified directory.

        Args:
            rel_path (str): The relative directory path to search within.
            pattern (str): The glob pattern to match files (e.g., '*.txt').

        Returns:
            List[str]: A list of relative file paths matching the pattern.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        full_path = self._full_path(rel_path)
        try:
            # Use glob for pattern matching
            matched_files = self.fs.glob(posixpath.join(full_path, pattern))
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory '{rel_path}' does not exist.")

        # Convert full paths back to relative paths
        relative_matched_files = [
            posixpath.relpath(file, self.root_path) for file in matched_files
        ]
        return relative_matched_files

    def copy(self, src_rel_path: str, dest_rel_path: str) -> None:
        """
        Copies a file from the source relative path to the destination relative path.

        Args:
            src_rel_path (str): The relative path to the source file.
            dest_rel_path (str): The relative path to the destination file.

        Raises:
            FileNotFoundError: If the source file does not exist.
        """
        src_full_path = self._full_path(src_rel_path)
        dest_full_path = self._full_path(dest_rel_path)
        self.fs.copy(src_full_path, dest_full_path)

        # Invalidate cache for destination file if it exists
        with self.lock:
            self.cache.pop(dest_rel_path, None)

    def mkdir(self, rel_path: str, create_parents: bool = True) -> None:
        """
        Creates a directory at the specified relative path.

        Args:
            rel_path (str): The relative path to the directory to create.
            create_parents (bool, optional): Whether to create parent directories if they do not exist.
                                             Defaults to True.
        """
        dir_full_path = self._full_path(rel_path)
        self.fs.makedirs(
            dir_full_path, exist_ok=True, mode=0o777, parents=create_parents
        )
        # No cache invalidation needed for directory creation

    def md5(self, rel_path: str, block_size: int = 65536) -> str:
        """
        Computes the MD5 checksum of the file at the specified relative path.

        Args:
            rel_path (str): The relative path to the file.
            block_size (int, optional): The size of each block read from the file. Defaults to 65536 bytes.

        Returns:
            str: The hexadecimal MD5 checksum of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._full_path(rel_path)
        hash_md5 = hashlib.md5()
        with self.fs.open(full_path, "r") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def rm(self, rel_path: str) -> None:
        """
        Removes the file at the specified relative path.

        Args:
            rel_path (str): The relative path to the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._full_path(rel_path)
        self.fs.rm(full_path)
        # Invalidate cache for the removed file
        with self.lock:
            self.cache.pop(rel_path, None)


def fs_read(rel_path: str, mode: str = "r", use_cache: bool = False):
    return session.get("filesystem", SessionFileSystem).read(
        rel_path, mode=mode, use_cache=use_cache
    )


def fs_write(rel_path: str, data: Any, mode: str = "w"):
    return session.get("filesystem", SessionFileSystem).write(rel_path, data, mode=mode)


def fs_ls(rel_path: str = ""):
    return session.get("filesystem", SessionFileSystem).ls(rel_path)


def fs_rm(rel_path: str):
    return session.get("filesystem", SessionFileSystem).rm(rel_path)


def fs_exists(rel_path: str = ""):
    return session.get("filesystem", SessionFileSystem).exists(rel_path)


def fs_glob(rel_path: str, pattern: str):
    return session.get("filesystem", SessionFileSystem).glob(rel_path, pattern)


def fs_copy(src_rel_path: str, dest_rel_path: str):
    return session.get("filesystem", SessionFileSystem).copy(
        src_rel_path, dest_rel_path
    )


def fs_mkdir(rel_path: str, create_parents: bool = True):
    return session.get("filesystem", SessionFileSystem).mkdir(
        rel_path, create_parents=create_parents
    )


def fs_md5(rel_path: str, block_size: int = 65536):
    return session.get("filesystem", SessionFileSystem).md5(
        rel_path, block_size=block_size
    )
