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

# Section for retrieving image zip folder
#------------------------------------------------
# Define the URL and file ID
print("Retrieving Image zip folder")
url = "https://drive.google.com/uc?export=download&id=176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"
file_id = "176wGCHHp2DpoDblsliEkX4fTpfQUbZOq"
# Define the output file path
output_file_path = "download_ds_file.zip"
# Download the file using wget command
subprocess.call(["wget", "-O", output_file_path, url])
print("Image zip folder retrieved")
#------------------------------------------------

# Section for retrieving text zip folder
#------------------------------------------------
# Define the URL and file ID
print("Retrieving Text zip folder")
url = "https://drive.google.com/uc?export=download&id=1sIxT8WrW21vaQvUY3BLGnnmAY-ocZhpO"
file_id = "1sIxT8WrW21vaQvUY3BLGnnmAY-ocZhpO"
# Define the output file path
output_file_path = "download_text_file.zip"
# Download the file using wget command
subprocess.call(["wget", "-O", output_file_path, url])
print("Text zip folder retrieved")
#------------------------------------------------

# Section for unziping text zip folder
#------------------------------------------------
#Required Libraries 
import zipfile

#Reading Caption Contents
def load_captions (filename):
    with open(filename, "r") as fp:
    # Read all text in the file
    text = fp.read()
    return (text)

# Extract the zip file
with zipfile.ZipFile(output_file_path, 'r') as zip_ref:
    extracted_captionfile_path = zip_ref.extract('Flickr8k.token.txt', path='.')

doc = load_captions(extracted_captionfile_path)

# Get current working directory
# cwd = os.getcwd()
# # Print current working directory
# print("Current working directory is:", cwd)





# Importing Libraries
# import tensorflow as tf
# import numpy as np
# import os
# import time
# import json
# from PIL import Image
# import pickle

## Data Preprocessing
# class Preprocessor:
#     def __init__(self, img_folder_path, annotation_file_path, max_vocab_size, max_length):
#         self.img_folder_path = img_folder_path
#         self.annotation_file_path = annotation_file_path
#         self.max_vocab_size = max_vocab_size
#         self.max_length = max_length

#     def get_image_names(self):
#         """
#         This function returns a list of all the image file names in the folder.
#         """
#         return os.listdir(self.img_folder_path)

#     def load_image(self, img_path):
#         """
#         This function loads the image from the given path and resizes it to the required size.
#         """
#         img = tf.io.read_file(self.img_folder_path + '/' + img_path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         img = tf.image.resize(img, (299, 299))
#         img = tf.keras.applications.inception_v3.preprocess_input(img)
#         return img, img_path

#     def load_annotations(self):
#         """
#         This function reads the annotations file and returns the image name and the corresponding caption.
#         """
#         with open(self.annotation_file_path, 'r') as f:
#             annotations = json.load(f)

#         # Store captions and image names in separate lists
#         all_captions = []
#         all_img_names = []

#         for annot in annotations['annotations']:
#             caption = '<startseq> ' + annot['caption'] + ' <endseq>'
#             img_name = annot['image_id']
#             all_img_names.append(img_name)
#             all_captions.append(caption)

#         return all_img_names, all_captions

#     def tokenize_captions(self, captions_list):
#         """
#         This function tokenizes the captions and returns the tokenizer along with the tokenized captions.
#         """
#         # Initialize the tokenizer
#         tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_vocab_size, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        
#         # Fit the tokenizer on the captions
#         tokenizer.fit_on_texts(captions_list)
        
#         # Tokenize the captions
#         tokenized_captions = tokenizer.texts_to_sequences(captions_list)
        
#         # Pad the tokenized sequences
#         captions_vector = tf.keras.preprocessing.sequence.pad_sequences(tokenized_captions, padding='post', maxlen=self.max_length)
        
#         return tokenizer, captions_vector

#     def preprocessed_data(self):
#         """
#         This function returns the preprocessed images and their corresponding captions.
#         """
#         # Get image names
#         all_img_names = self.get_image_names()
        
#         # Get image features
#         image_features_extract_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
#         new_input = image_features_extract_model.input
#         hidden_layer = image_features_extract_model.layers[-1].output
#         image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
#         img_features = {}

#         for count, img_name in enumerate(all_img_names):
#             img, path = self.load_image(img_name)
#             batch_features = image_features_extract_model(img)
#             batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
#             img_features[path] = batch_features.numpy()

#             if count % 1000 == 0:
#                 print(f"Processed {count} images")

        # Save image