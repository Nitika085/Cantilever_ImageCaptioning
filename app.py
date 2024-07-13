from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('models/image_captioning_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load InceptionV3 model
image_model = InceptionV3(weights='imagenet')
image_model = tf.keras.Model(image_model.input, image_model.layers[-2].output)

max_length = 34  # This should match what you used during training

def generate_caption(image):
    # Preprocess the image
    image = tf.image.resize(image, (299, 299))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    # Extract features
    features = image_model.predict(image, verbose=0)
    
    # Generate caption
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, '')
        if word == '' or word == 'endseq':
            break
        in_text += ' ' + word
    
    return in_text.replace('startseq', '').strip()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            image = tf.image.decode_image(file.read(), channels=3)
            caption = generate_caption(image)
            return render_template('result.html', caption=caption)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
