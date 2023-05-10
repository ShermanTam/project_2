import subprocess
import zipfile

# Define the URL and file ID
url = "https://drive.google.com/uc?export=download&id=176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"
file_id = "176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"

# Define the output file path
# Replace "path/to/folder" with the actual path to your folder
output_file_path = "/Users/varshasrinivas/Desktop/SJSU/SPRING 2023/Deep Learning/Project/ImageCaptionGenerator/dataset_folder/download_img.zip"

# Download the file using wget command
subprocess.call(["wget", "-O", output_file_path, url])

# Unzip the downloaded file
with zipfile.ZipFile(output_file_path, "r") as zip_ref:
    # Replace "path/to/folder" with the actual path to your folder
    zip_ref.extractall(
        "/Users/varshasrinivas/Desktop/SJSU/SPRING 2023/Deep Learning/Project/ImageCaptionGenerator/dataset_folder")

# Use the extracted dataset in your attention model code
# ...
