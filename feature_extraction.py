import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import cv2

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

def process_images(image_directory):
    filenames = []
    feature_list = []
    product_ids = []
    
    # Create a list of image files
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                filepath = os.path.join(root, file)
                try:
                    # Extract product ID from filename (assuming format: id_*.jpg)
                    product_id = int(os.path.basename(file).split('_')[0])
                    
                    # Extract features
                    features = extract_features(filepath, model)
                    
                    filenames.append(filepath)
                    feature_list.append(features)
                    product_ids.append(product_id)
                    print(f"Processed {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")
    
    return filenames, feature_list, product_ids

def main():
    # Directory containing your product images
    image_directory = 'product_images'
    
    # Ensure the directory exists
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
        print(f"Created directory: {image_directory}")
        print("Please add your product images to this directory and run the script again.")
        print("Image filename format should be: productID_description.jpg")
        return
    
    print("Starting feature extraction...")
    filenames, feature_list, product_ids = process_images(image_directory)
    
    if not filenames:
        print("No images found in the product_images directory.")
        return
    
    # Save the results
    feature_list = np.array(feature_list)
    
    pickle.dump(feature_list, open('featurevector.pkl', 'wb'))
    pickle.dump(filenames, open('filenames.pkl', 'wb'))
    pickle.dump(product_ids, open('product_ids.pkl', 'wb'))
    
    print("Feature extraction complete!")
    print(f"Processed {len(filenames)} images")
    print("Created: featurevector.pkl, filenames.pkl, product_ids.pkl")

if __name__ == "__main__":
    main()
