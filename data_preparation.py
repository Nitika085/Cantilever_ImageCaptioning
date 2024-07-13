import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer

def load_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def load_captions(caption_file):
    with open(caption_file, 'r', encoding='utf-8') as f:
        captions = f.read().split('\n')
    
    caption_dict = {}
    for i, caption in enumerate(captions, 1):
        caption = caption.strip()
        if not caption:  # Skip empty lines
            continue
        try:
            img, cap = caption.split(',', 1)
            if img not in caption_dict:
                caption_dict[img] = []
            caption_dict[img].append('startseq ' + cap.strip() + ' endseq')
        except ValueError:
            print(f"Warning: Line {i} is not in the expected format: {caption}")
            continue
    
    if not caption_dict:
        raise ValueError("No valid captions were found in the file.")
    
    return caption_dict

def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def prepare_data(image_dir, caption_file):
    # ... (previous code remains the same)
    
    image_paths = []
    all_captions = []
    
    for img, caps in captions.items():
        img_path = os.path.join(image_dir, img)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            all_captions.extend(caps)
        else:
            print(f"Warning: Image file not found: {img_path}")
    
    if not image_paths:
        raise ValueError("No valid images were found.")
    
    tokenizer = tokenize_captions(all_captions)
    max_length = max(len(caption.split()) for caption in all_captions)
    
    print(f"Processed {len(image_paths)} images and {len(all_captions)} captions.")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Maximum caption length: {max_length}")
    
    return image_paths, all_captions, tokenizer, max_length
    

if __name__ == "__main__":
    # Example usage
    image_dir = 'path/to/your/Flickr8k_Dataset'
    caption_file = 'path/to/your/Flickr8k_text/captions.txt'
    try:
        images, captions, tokenizer, max_length = prepare_data(image_dir, caption_file)
        print("Data preparation completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    