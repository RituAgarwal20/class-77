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

with open('../features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('../captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()


model = keras.models.load_model('../best_model.h5')

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

for key, captions in mapping.items():
    for i in range(len(captions)):
        caption = captions[i]
        caption = caption.lower()
        caption = caption.replace('[^a-z]', '')
        caption = caption.replace('\s+', ' ')
        caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
        captions[i] = caption

print("::::::::::::::::::::::::Mapping after cleaning::::::::::::::::::::::::")
print(mapping["1000268201_693b08cb0e"])

# Create an empty list for storing all_captions
all_captions = []
# Run a loop for each key in the mapping
for key in mapping:
    # Run loop for each caption in the mapping[key]
    for caption in mapping[key]:
        # Append caption to all_captions
        all_captions.append(caption)

# Create a tokenizer
tokenizer = Tokenizer()
# Updates internal vocabulary of tokenizer based on a list of texts.
tokenizer.fit_on_texts(all_captions)

max_length = max(len(caption.split()) for caption in all_captions)
max_length

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer):
    global max_length
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
      
    return in_text

vgg_model = VGG16()

vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

cap = cv2.VideoCapture("../video3.mp4")

i=100
while True:
    try:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if i == 100:
            image = np.asarray(img)     
            image = img
            image = cv2.resize(image, (224, 224))
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = vgg_model.predict(image, verbose=0)
            
            # Call the predict_caption function with model, feature, tokenizer and save the returned value in variable caption
            caption = predict_caption(model, feature, tokenizer)
            
            caption = caption.replace('startseq ', '')
            caption = caption.replace('endseq', '')

            i = 0
                    
        img = cv2.putText(img, caption, (20,50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        print(caption)
        i = i+1      

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Exception", e)
cap.release()
cv2.destroyAllWindows()

