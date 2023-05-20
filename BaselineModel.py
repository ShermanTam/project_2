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

print("Epoch number:", os.environ.get('EPOCH_NUMBER'))
print("Batch number:", os.environ.get('BATCH_SIZE'))
print("Model Type:",  os.environ.get('MODEL_TYPE'))

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
    
                # return(content)

        mapping = {}
        for line in tqdm(content.split('\n')):
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            caption = " ".join(caption)
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)

        self.mapping = mapping

    def clean_captions(self):
        for key, captions in self.mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                # Remove punctuation marks
                caption = "".join([c if c.isalpha() else " " for c in caption])
                # Remove extra whitespaces
                caption = " ".join(caption.split())
                # Update the cleaned caption
                captions[i] = caption

    def create_tokenizer(self):
        all_captions = []
        for key, captions in self.mapping.items():
            all_captions.extend(captions)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1

    def create_sequences(self):
        sequences = []
        for key, captions in self.mapping.items():
            for caption in captions:
                # Convert caption to sequence of integers
                sequence = self.tokenizer.texts_to_sequences([caption])[0]
                # Generate multiple input-output pairs for the sequence
                for i in range(1, len(sequence)):
                    in_seq, out_seq = sequence[:i], sequence[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    sequences.append((key, in_seq, out_seq))

        return sequences

    def generate_data(self, sequences):
        X_image, X_sequence, y = [], [], []
        for key, in_seq, out_seq in sequences:
            # Retrieve image features
            image_feature = self.features[key][0]
            # Append the data to respective lists
            X_image.append(image_feature)
            X_sequence.append(in_seq)
            y.append(out_seq)

        return np.array(X_image), np.array(X_sequence), np.array(y)

    def define_model(self):
        # Image feature input
        inputs1 = Input(shape=(4096,))
        x1 = Dropout(0.5)(inputs1)
        x2 = Dense(256, activation='relu')(x1)

        # Sequence input
        inputs2 = Input(shape=(self.max_length,))
        y1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        y2 = Dropout(0.5)(y1)
        y3 = LSTM(256)(y2)

        # Decoder model
        decoder1 = add([x2, y3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        # Merge the models
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        self.model = model

    def generate_caption(self, photo):
        # Preprocess the image
        img = load_img(photo, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)

        # Extract features from the image
        feature = self.model.predict(img, verbose=0)

        # Generate a sequence of integers
        sequence = [self.tokenizer.word_index['startseq']]

        # Generate the caption word by word
        for _ in range(self.max_length):
            # Pad the sequence
            sequence = pad_sequences([sequence], maxlen=self.max_length)

            # Predict the next word
            yhat = self.model.predict([feature, sequence], verbose=0)

            # Convert probability to integer index
            yhat = np.argmax(yhat)

            # Map integer index to word
            word = self.tokenizer.index_word[yhat]

            # Stop if we predict the end of the sequence
            if word == 'endseq':
                break

            # Append the predicted word to the sequence
            sequence[0].append(yhat)

        # Remove start and end tokens from the sequence
        caption = sequence[0][1:-1]

        # Convert the sequence of integers to a caption
        caption = [self.tokenizer.index_word[word] for word in caption]
        caption = ' '.join(caption)

        return caption

# Instantiate the ImageCaptionGenerator
generator = ImageCaptionGenerator()

print("Loading VGG16 model")
generator.extract_image_features()

print("Extracting image features")
generator.load_image_features("datasets/Flicker8k_Dataset")

print("Loading captions data from the dataset file")
generator.load_captions_data("datasets/download_ds_file.zip","Flickr8k.token.txt")

print("Preprocessing the captions")
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
print("Training the model")
generator.model.fit([X_image_train, X_sequence_train], y_train, batch_size=32, epochs=10, verbose=1,
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