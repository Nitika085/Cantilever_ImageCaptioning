import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import pickle
from data_preparation import prepare_data
from model import create_model

def load_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def train_model(image_paths, captions, tokenizer, max_length, epochs=10, batch_size=16):
    vocab_size = len(tokenizer.word_index) + 1
    model, image_model = create_model(vocab_size, max_length)
    
    def data_generator(image_paths, captions, tokenizer, max_length, batch_size):
        n = len(image_paths)
        while True:
            for i in range(0, n, batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_captions = captions[i:i+batch_size]
                
                batch_images = np.array([load_image(path) for path in batch_paths])
                batch_features = image_model.predict(batch_images, verbose=0)
                
                input_seq = tokenizer.texts_to_sequences(batch_captions)
                input_seq = pad_sequences(input_seq, maxlen=max_length)
                
                output_seq = to_categorical(input_seq, num_classes=vocab_size)
                
                yield [[batch_features, input_seq[:, :-1]], output_seq[:, 1:]]
    
    steps = len(image_paths) // batch_size
    model.fit(data_generator(image_paths, captions, tokenizer, max_length, batch_size),
              steps_per_epoch=steps,
              epochs=epochs,
              verbose=1)
    
    # Save the model and tokenizer
    model.save('models/image_captioning_model.h5')
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return model, tokenizer

if __name__ == "__main__":
    image_dir = 'data/Flickr8k_Dataset'
    caption_file = 'data/Flickr8k_text/captions.txt'
    
    image_paths, captions, tokenizer, max_length = prepare_data(image_dir, caption_file)
    model, tokenizer = train_model(image_paths, captions, tokenizer, max_length)
    
    print("Training completed successfully.")