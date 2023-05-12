import urllib.request

# # URL of the Flickr8k dataset folder or file
# dataset_url = "https://drive.google.com/drive/folders/1WNHl00Xuxh8-R2-VpJR5GLKszBkCOX83?usp=share_link"

# Receiving Input variables from Github Repository Action
#------------------------------------------------
import os
print("Epoch number:", os.environ.get('EPOCH_NUMBER'))
print("Batch number:", os.environ.get('BATCH_SIZE'))
print("Model Type:",  os.environ.get('MODEL_TYPE'))
#------------------------------------------------

#--------------------------------------------------
# The Flickr8k dataset have two main zip files- Images zip file & Captions zip file
# Retrieving the Flickr8k dataset folder using the file ids
# {"image_name_1" : ["caption 1", "caption 2", "caption 3"],
#  "image_name_2" : ["caption 4", "caption 5"]}
#--------------------------------------------------

#Required Libraries 
import subprocess

# Section for unziping text zip folder
#-----------------------------------------------------------       
#Required Libraries 
import zipfile
#----------------------------------------------------------- 
# 
#
#
#-----------------------------------------------------------       

def load_captions(zip_file_path,file_to_access):
    # Extract the file from the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with zip_ref.open(file_to_access) as file:
            content = file.read()
            return(content)
#-----------------------------------------------------------       

#-----------------------------------------------------------       
# Each photo has a unique identifier, which is the file name of the image .jpg file
# Create a dictionary of photo identifiers (without the .jpg) to captions. Each photo identifier maps to
# a list of one or more textual descriptions.
#
# {"image_name_1" : ["caption 1", "caption 2", "caption 3"],
#  "image_name_2" : ["caption 4", "caption 5"]}
#-----------------------------------------------------------       
def captions_dict (text):
  dict = {}
  ## Converting Bytes to string
  text = text.decode('utf-8')
  # Make a List of each line in the file
  lines = text.split ('\n')
  for line in lines:
    # Split into the <image_data> and <caption>
    line_split = line.split ('\t')
    if (len(line_split) != 2):
      # Added this check because dataset contains some blank lines
      continue
    else:
      image_data, caption = line_split
    # Split into <image_file> and <caption_idx>
    image_file, caption_idx = image_data.split ('#')
    # Split the <image_file> into <image_name>.jpg
    image_name = image_file.split ('.')[0]
    # If this is the first caption for this image, create a new list for that
    # image and add the caption to it. Otherwise append the caption to the 
    # existing list
    if (int(caption_idx) == 0):
      dict [image_name] = [caption]
    else:
      dict [image_name].append (caption)
  return (dict)

print("Retrieving text files from zip folder")
doc = load_captions ("download_ds_file.zip","Flickr8k.token.txt")
image_dict = captions_dict (doc)
#-----------------------------------------------------------       

#-----------------------------------------------------------       
# We have three separate files which contain the names for the subset of 
# images to be used for training, validation or testing respectively
#
# Given a file, we return a set of image names (without .jpg extension) in that file
#-----------------------------------------------------------
def subset_image_name (train_img_txt):
  data = []
  ## Converting Bytes to string
  train_img_txt = train_img_txt.decode('utf-8')
  # Make a List of each line in the file
  lines = train_img_txt.split ('\n')
  for line in lines:
        # skip empty lines
        if (len(line) < 1):
              continue
        # Each line is the <image_file>
        # Split the <image_file> into <image_name>.jpg
        image_name = line.split ('.')[0]
        # Add the <image_name> to the list
        data.append (image_name)

  return (set(data))  

print("Retrieving names of training images from text file")
training_imgname_doc = load_captions("download_ds_file.zip","Flickr_8k.trainImages.txt")
training_image_names = subset_image_name (training_imgname_doc)
# print(training_image_names)
#-----------------------------------------------------------       

#-----------------------------------------------------------
# Clean the captions data
#    Convert all words to lowercase.
#    Remove all punctuation.
#    Remove all words that are one character or less in length (e.g. ‘a’).
#    Remove all words with numbers in them.
#-----------------------------------------------------------
## Required Libraries
import re
def captions_clean (image_dict):
      print_count=0
      # <key> is the image_name, which can be ignored
      for key, captions in image_dict.items():
            # Loop through each caption for this image
            for i, caption in enumerate (captions):
                  # Convert the caption to lowercase, and then remove all special characters from it
                  caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
                  # Split the caption into separate words, and collect all words which are more than 
                  # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
                  clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
                  # Join those words into a string
                  caption_new = ' '.join(clean_words)
                  if print_count<=10:
                        print("\t Old caption:",captions[i])
                  
                  # Replace the old caption in the captions list with this new cleaned caption
                  captions[i] = caption_new
                  if print_count<=10:
                        print("\t New caption:",captions[i])
                  
                  print_count += 1
#-----------------------------------------------------------
print("Preprocessing captions:")
captions_clean (image_dict)
#-----------------------------------------------------------

#-----------------------------------------------------------
# Load images
#-----------------------------------------------------------
## Required Libraries
# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm

print("Extracting images:")

# with zipfile.ZipFile('download_img_file.zip', 'r') as zip_ref:
#     zip_ref.extractall('extracted_images')
#     print("zip file extracted")


# zip_path = "extracted_images.zip"  # Path to the extracted zip file

# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     # List all the files in the zip
#     file_list = zip_ref.namelist()

#     # Access individual files in the zip
#     for file_name in file_list:
#         print(file_name)

# import zipfile

# Specify the path to the downloaded ZIP file
# zip_path = 'download_img_file.zip'

# # Specify the directory where you want to extract the images
# extract_dir = 'extracted_images'

# # Extract the ZIP file
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)

# print('ZIP file extracted.')

import requests
import zipfile
import io

# def download_and_extract(url, dest):
#     response = requests.get(url)
#     zipped_file = zipfile.ZipFile(io.BytesIO(response))
#     zipped_file.extractall(dest)
#     zipped_file.close()


# zip_url = "https://drive.google.com/uc?export=download&id=176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"
# extract_to_destination = "extracted_images"

# download_and_extract(zip_url, extract_to_destination)

# Access the extracted files
# For example, you can list the files inside the extracted folder

# Extract the file from the zip file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#   with zip_ref.open(file_to_access) as file:
#     content = file.read()
#     return(content)
# import zipfile

def get_images(image_zip_filepath):
      with zipfile.ZipFile(image_zip_filepath,'r') as zip_ref:
        file_names = [name for name in zip_ref.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        return(file_names)

image_zip_filepath="download_image_file.zip"
images=get_images(image_zip_filepath)
print(images)