import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import cv2
from nltk.translate.bleu_score import corpus_bleu
import zipfile
from sklearn.model_selection import train_test_split


class ImageCaptionGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.vocab_size = None
        self.features = None
        self.mapping = None

    def extract_image_features(self):
        # Load VGG16 model
        model = VGG16()
        # Restructure the model
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        self.model = model

    def load_image_features(self, folder_path):
        file_names = os.listdir(folder_path)
        features = {}  # Initialize an empty dictionary to store image features
        for file in tqdm(file_names):
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path)
            if img is None:
                continue  # Skip this image if it couldn't be loaded

            target_size = (224, 224)
            img = cv2.resize(img, target_size)
            image = img_to_array(img)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = self.model.predict(image, verbose=0)
            image_id = file.split('.')[0]
            features[image_id] = feature

        print("Features extracted:",features)
        self.features = features

    def save_image_features(self, file_path):
        pickle.dump(self.features, open(file_path, 'wb'))

    def load_image_features_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            self.features = pickle.load(f)


    def load_captions_data(self,zip_file_path,file_to_access):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            with zip_ref.open(file_to_access) as file:
                content = file.read().decode('utf-8')
print("Loading VGG16 model")
generator.extract_image_features()

print("Extracting image features")
generator.load_image_features("datasets/Flicker8k_Dataset")

print("Loading captions data from the dataset file")
generator.load_captions_data("datasets/download_ds_file.zip","Flickr8k.token.txt")

print("Preprocessing the captions:")
generator.clean_captions()

print("\t tokenization and other preprocessing") 
generator.create_tokenizer()

# Set the maximum length for sequences
generator.max_length = 20

# Create sequences from the captions
sequences = generator.create_sequences()

# Generate data for model training
X_image, X_sequence, y = generator.generate_data(sequences)

# Split the data into train, test, and validation sets
print("Splitting the data into train, test, and validation sets")
X_image_train, X_image_test, X_sequence_train, X_sequence_test, y_train, y_test = train_test_split(
    X_image, X_sequence, y, test_size=0.2, random_state=42)
X_image_train, X_image_val, X_sequence_train, X_sequence_val, y_train, y_val = train_test_split(
    X_image_train, X_sequence_train, y_train, test_size=0.1, random_state=42)

# Define the model
generator.define_model()


# Train the model using the generated data
print("Training the model:")


print("\t Epoch number:", os.environ.get('EPOCH_NUMBER'))
print("\t Batch number:", os.environ.get('BATCH_SIZE'))

generator.model.fit([X_image_train, X_sequence_train], y_train, batch_size=os.environ.get('BATCH_SIZE'), epochs=os.environ.get('EPOCH_NUMBER'), verbose=1,
                    validation_data=([X_image_val, X_sequence_val], y_val))
generator.model.fit([X_image_train, X_sequence_train], y_train, 
                    validation_data=([X_image_val, X_sequence_val], y_val),
                    epochs=10, batch_size=64)
print("Model trained for the specified number of epochs and batch size")

# Evaluate the model on the test set
print("Evaluating the model on the test set")
generator.model.evaluate([X_image_test, X_sequence_test], y_test, verbose=1)
test_loss = generator.model.evaluate([X_image_test, X_sequence_test], y_test)
print("Test Loss:", test_loss)