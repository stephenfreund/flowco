import sys
import boto3
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta


def format_relative_time(mod_time):
    """Converts a modification datetime into a relative time string."""
    now = datetime.now(timezone.utc)
    delta = relativedelta(now, mod_time)

    if delta.years > 0:
        return f"{delta.years} year{'s' if delta.years != 1 else ''} ago"
    elif delta.months > 0:
        return f"{delta.months} month{'s' if delta.months != 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
    elif delta.hours > 0:
        return f"{delta.hours} hour{'s' if delta.hours != 1 else ''} ago"
    elif delta.minutes > 0:
        return f"{delta.minutes} minute{'s' if delta.minutes != 1 else ''} ago"
    else:
        return "Just now"


def get_bucket_info(bucket_name, show_files=False):
    """
    Retrieves all objects from the given bucket and groups them by the top-level directory.
    Each directory record includes the latest modification time among its files.
    If show_files is True, each directory record will also include a list of files with their modification times.
    """
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name)

    if "Contents" not in response:
        print("No files found in the bucket.")
        return {}

    directories = {}

    for obj in response["Contents"]:
        key = obj["Key"]
        mod_time = obj["LastModified"]

        # Determine the directory name (if there is no "/", consider it "root")
        parts = key.split("/", 1)
        if len(parts) > 1:
            directory = parts[0]
            file_name = parts[1]
        else:
            directory = "root"
            file_name = key

        # Initialize the directory record if it doesn't exist
        if directory not in directories:
            directories[directory] = {"last_modified": mod_time, "files": []}
        else:
            # Update the directory's latest modification time if necessary
            if mod_time > directories[directory]["last_modified"]:
                directories[directory]["last_modified"] = mod_time

        if show_files:
            directories[directory]["files"].append((file_name, mod_time))

    return directories


def main():
    bucket_name = "go-flowco"
    show_files = False

    # Check for the '-a' flag in the arguments
    args = sys.argv[1:]
    if "-a" in args:
        show_files = True
        args.remove(
            "-a"
        )  # Remove the flag so any additional args can be processed later if needed

    directories = get_bucket_info(bucket_name, show_files)
    if directories:
        # Sort directories by last modified time (most recent first)
        sorted_dirs = sorted(
            directories.items(), key=lambda x: x[1]["last_modified"], reverse=True
        )
        for directory, info in sorted_dirs:
            relative_time = format_relative_time(info["last_modified"])
            print(f"{directory:40}{relative_time:20} {info['last_modified']}")
            # If the -a flag is provided, list each file under the directory
            if show_files and info["files"]:
                # Sort files by modification time (most recent first)
                sorted_files = sorted(info["files"], key=lambda x: x[1], reverse=True)
                for file_name, mod_time in sorted_files:
                    file_relative = format_relative_time(mod_time)
                    print(f"    {file_name:36}{file_relative:20} {mod_time}")
    else:
        print("No directories found.")


if __name__ == "__main__":
    main()
