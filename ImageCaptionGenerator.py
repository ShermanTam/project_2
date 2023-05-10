import urllib.request

# # URL of the Flickr8k dataset folder or file
# dataset_url = "https://example.com/path/to/flickr8k_dataset.zip"

# # Define the local path to save the dataset in the GitHub repository workspace
# local_path = "/path/in/github/repository/flickr8k_dataset.zip"

# # Download the dataset file from the online source
# urllib.request.urlretrieve(dataset_url, local_path)

import subprocess

# Define the URL and file ID
url = "https://drive.google.com/uc?export=download&id=176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"
file_id = "176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"

# Define the output file path
output_file_path = "download_ds_file.zip"

# Download the file using wget command
subprocess.call(["wget", "-O", output_file_path, url])
