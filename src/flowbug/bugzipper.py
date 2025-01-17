import boto3
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import os
import zipfile
from io import BytesIO


def parse_relative_time(relative_time_str):
    # Split the relative time string (e.g., "2 days ago")
    parts = relative_time_str.split()
    if len(parts) != 3 or parts[2] != "ago":
        raise ValueError(f"Invalid relative time format: {relative_time_str}")

    # Convert the numeric value and time unit
    value = int(parts[0])
    unit = parts[1].lower()

    # Calculate the current time minus the relative delta
    now = datetime.now(timezone.utc)

    if unit in ["day", "days"]:
        return now - relativedelta(days=value)
    elif unit in ["week", "weeks"]:
        return now - relativedelta(weeks=value)
    elif unit in ["month", "months"]:
        return now - relativedelta(months=value)
    elif unit in ["year", "years"]:
        return now - relativedelta(years=value)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")


def get_files_after_time(bucket_name, relative_time_str):
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Parse the relative time to get the cutoff datetime
    cutoff_time = parse_relative_time(relative_time_str)

    # Get the list of all objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)

    # Check if the bucket is empty
    if "Contents" not in response:
        print("No files found in the bucket.")
        return []

    # Filter the files based on the creation date and file extension
    zip_files = []
    for obj in response["Contents"]:
        last_modified = obj["LastModified"]
        if last_modified >= cutoff_time and obj["Key"].endswith(".zip"):
            zip_files.append({"Key": obj["Key"], "LastModified": last_modified})

    # Sort files by LastModified in descending order (most recent first)
    zip_files.sort(key=lambda x: x["LastModified"], reverse=True)

    return zip_files


def download_zip_files(bucket_name, zip_files, download_dir):
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Track downloaded files and descriptions
    downloaded_files = []
    descriptions = []

    # Download the files
    for zip_file in zip_files:
        key = zip_file["Key"]
        # Extract the parent directory name
        parent_dir = key.split("/")[0]

        # Get the base name of the zip file
        file_name = os.path.basename(key)

        # Construct the new file name with parent directory included
        new_file_name = f"{parent_dir}_{file_name}"

        # Define the download path
        download_path = os.path.join(download_dir, new_file_name)

        # Skip downloading if the file already exists
        if os.path.exists(download_path):
            print(f"Skipping {key}, file already exists.")
        else:
            # Download the zip file
            print(f"Downloading {key} to {download_path}...")
            s3.download_file(bucket_name, key, download_path)
            downloaded_files.append(new_file_name)

        # Extract the description.txt from the zip file, regardless of download
        with BytesIO() as file_stream:
            s3.download_fileobj(bucket_name, key, file_stream)
            with zipfile.ZipFile(file_stream, "r") as zip_ref:
                # Check if description.txt exists in the zip file
                if "description.txt" in zip_ref.namelist():
                    with zip_ref.open("description.txt") as description_file:
                        descriptions.append(
                            (new_file_name, description_file.read().decode("utf-8"))
                        )
                else:
                    descriptions.append((new_file_name, "No description.txt found"))

    return downloaded_files, descriptions


def main():
    # Bucket name and relative time string (e.g., "2 days ago")
    bucket_name = "go-flowco"
    relative_time_str = "2 days ago"  # Specify your relative time

    # Get the list of zip files created after the given time
    zip_files = get_files_after_time(bucket_name, relative_time_str)

    if zip_files:
        print(f"Found {len(zip_files)} zip files to download.")

        # Download the files and extract descriptions
        download_dir = "./downloads"
        downloaded_files, descriptions = download_zip_files(
            bucket_name, zip_files, download_dir
        )

        if downloaded_files:
            print("\nFiles downloaded:")
            for file in downloaded_files:
                print(file)

        print("\nDescriptions:")
        for file_name, description in descriptions:
            print(f"\n{file_name}:")
            print(description)
    else:
        print("No zip files found for the specified time range.")


if __name__ == "__main__":
    main()
