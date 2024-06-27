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

# load features from features.pkl
with open('../features.pkl', 'rb') as f:
    features = pickle.load(f)

# Load the captions from caption.txt
with open('../captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

# Load the model
model = keras.models.load_model('../best_model.h5')

# Create mapping of image to captions
mapping = {}
# Loop through every caption
for line in captions_doc.split('\n'):
    # Split the line by comma(,)
    tokens = line.split(',')
    # Move to next iteration if length of line is less then 2 characters
    if len(line) < 2:
        continue
    # Take image_id and caption from token[0], [1] respectively
    image_id, caption = tokens[0], tokens[1:]
    # Remove extension from image ID
    image_id = image_id.split('.')[0]
    # Convert caption list to string
    caption = " ".join(caption)
    # Create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # Store the caption
    mapping[image_id].append(caption)

# Print the mapping dictionary
print(mapping["1000268201_693b08cb0e"])



