# -*- coding: utf-8 -*-
"""ImageCaptionGenerator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y1qZDHFw88kgIm6E59hnAbXWkiZxqoD3

#### Image caption generator using flickr dataset
#### CNN for image feature extraction and LSTM forNLP

##### Importing modules
"""

import os ##hanadling files 
import pickle ##storing numpy features - ex. storing img features
import numpy as np ##basic
from tqdm.notebook import tqdm ##for UI for getting how much data is processed

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input ##extracting features from image data & preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array ##
from tensorflow.keras.preprocessing.text import Tokenizer ##for preprocessing text 
from tensorflow.keras.preprocessing.sequence import pad_sequences ##evening out whole text representation of sequence ( some will have 10 some will have 5)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model ##utilities -- representation of whole model as images
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add ##importing layers --

"""##### loading data from google drive and copying the contents to another folder(images)"""

from google.colab import drive
drive.mount('/content/drive')

# !unzip -uq "/content/drive/My Drive/Images.zip" -d "/content/drive/My Drive/captiongenerationimages"

"""##### Extract Image Features"""

# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output) ##we dont need the fully connected layer of VGG16, only previous layers to get feature results (feature extraction). thats why leaving the last layer and getting before layer and assigning to output
# summarize
print(model.summary())

import os
folder_path = '/content/drive/MyDrive/captiongenerationimages'
if os.path.exists(folder_path):
    print("Folder exists!")
else:
    print("Folder does not exist!")

import cv2
# extract features from image
features = {}
os.chdir("/content/drive/MyDrive/captiongenerationimages")
for file in os.listdir():
  img = cv2.imread(file)
  # Target size
  target_size = (224,224)
  # Resize image
  img = cv2.resize(img, target_size)
  # convert image pixels to numpy array
  image = img_to_array(img)
  # reshape data for model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # preprocess image for vgg
  image = preprocess_input(image)
  # extract features
  feature = model.predict(image, verbose=0)
  # get image ID
  image_id = file.split('.')[0]
  # store feature
  features[image_id] = feature

WORKING_DIR = '/content/drive/MyDrive/'

# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# print(features)

"""##### Commenting this part"""

# BASE_DIR = WORKING_DIR = '/content/drive/MyDrive/'

##!unzip -uq "/content/drive/My Drive/captions.txt.zip"

# os.chdir("/content/drive/MyDrive/")

# ls

"""Loading the Captions Data"""

BASE_DIR = WORKING_DIR = '/content/drive/MyDrive/'

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)

len(mapping)

"""Preprocessing Text Data

"""

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

# before preprocess of text
mapping['1000268201_693b08cb0e']

# preprocess the text
clean(mapping)

# after preprocess of text
mapping['1000268201_693b08cb0e']

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

len(all_captions)

all_captions[:15]

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

vocab_size

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length

"""Train Test Split

"""

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# startseq girl going into wooden building endseq
#        X                   y
# startseq                   girl
# startseq girl              going
# startseq girl going        into
# ...........
# startseq girl going into wooden building      endseq

# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)

# train the model
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# save the model
model.save(WORKING_DIR+'/best_model.h5')

"""Generate Captions for the Image

"""

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length) 
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
    
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

"""Visualize the Results"""

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "captiongenerationimages", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)

generate_caption("1002674143_1b742ab4b8.jpg")

print(y_pred)

generate_caption("1001773457_577c3a7d70.jpg")

"""Audio Generation"""

from tensorflow.keras.models import load_model
model = load_model(WORKING_DIR + '/best_model.h5')

pip install pyttsx3

pip uninstall pyttsx3

!apt-get update
!apt-get install espeak

import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set the speech rate
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50)

# Convert the caption to speech
caption_text = generate_caption("1001773457_577c3a7d70.jpg")
engine.say(caption_text)
engine.runAndWait()

import os

text = "Hello, this is a test message."
os.system(f"espeak '{text}'")

"""##### Another way"""

pip install gTTS

# Import the required module for text 
# to speech conversion
from gtts import gTTS
  
# This module is imported so that we can 
# play the converted audio
import os
  
# The text that you want to convert to audio
mytext = 'Welcome to geeksforgeeks!'
  
# Language in which you want to convert
language = 'en'
  
# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("image.mp3")

# Playing the converted file
os.system("mpg321 welcome.mp3")

# Import the required module for text
# to speech conversion
from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os

# The text that you want to convert to audio
mytext = 'Welcome to geeksforgeeks!'

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome
myobj.save("welcome.mp3")

# Playing the converted file
os.system("mpg321 welcome.mp3")
