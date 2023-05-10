import urllib.request

# URL of the Flickr8k dataset folder or file
dataset_url = "https://example.com/path/to/flickr8k_dataset.zip"

# Define the local path to save the dataset in the GitHub repository workspace
local_path = "/path/in/github/repository/flickr8k_dataset.zip"

# Download the dataset file from the online source
urllib.request.urlretrieve(dataset_url, local_path)
