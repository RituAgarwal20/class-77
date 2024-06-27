import os
import pickle
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow import keras

import cv2
import numpy as np

features = 'null'

with open('.\features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('.\captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()


model = keras.models.load_model('.\best_model.h5')

mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

print(mapping["1000268201_693b08cb0e"])

# Loop throught each key and captions in the mapping
for key, captions in mapping.items():
    # Go throught each caption in captions for a given image
    for i in range(len(captions)):
        # Take one caption at a time
        caption = captions[i]
        # Convert to lowercase
        caption = caption.lower()
        # Delete digits, special chars, etc., 
        caption = caption.replace('[^a-z]', '')
        # Delete additional spaces
        caption = caption.replace('\s+', ' ')
        # Add start and end tags to the caption
        caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
        # Store the cleaned caption back
        captions[i] = caption

# Print the cleaned mapping
print("::::::::::::::::::::::::Mapping after cleaning::::::::::::::::::::::::")
print(mapping["1000268201_693b08cb0e"])

