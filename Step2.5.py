# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:49:35 2024

@author: beros
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:49:35 2024

@author: beros
"""
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf 
import os
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow import io
from warnings import filterwarnings
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt 

#Load model (from parts 1-4)
model_path =r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\AER850_Project2.h5"
model = tf.keras.models.load_model(model_path)

#Process and predict images
def process_and_predict(image_path, model):
    img = image.load_img(image_path, target_size=(500, 500))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_probs = predictions[0]
    predicted_class_index = np.argmax(class_probs)
    predicted_probability = class_probs[predicted_class_index]
    
    return predicted_class_index, predicted_probability, class_probs

#Path to test images
test_images = {
    "crack": r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\Data\test\crack\test_crack.jpg",
    "missing-head":r"C:\Users\beros\OneDrive\Desktop\test_missinghead.jpg",
    "paint-off": r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\Data\test\paint-off\test_paintoff.jpg"
}

class_labels = {0: "crack", 1: "missing-head", 2: "paint-off"} #from model

#Visualize
def display_prediction(img_path, actual_label, predicted_label, class_probs, class_labels):
    img = image.load_img(img_path, target_size=(500, 500))
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True Label: {actual_label}\nPredicted Label: {predicted_label}", fontsize=14)
    prob_text = " ".join([f"{class_labels[i]}: {class_probs[i]*100:.1f}%" for i in range(len(class_probs))])
    plt.text(10, 430, prob_text, fontsize=16, color='green', va='top')
    plt.show()

#Display predictions
for label, img_path in test_images.items():
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at {img_path}")
        continue
  
    try:
        predicted_class_index, predicted_probability, class_probs = process_and_predict(img_path, model)
        predicted_label = class_labels[predicted_class_index]
        display_prediction(img_path, label, predicted_label, class_probs, class_labels)
    except Exception as e:
        print(f"Error processing {label} image: {str(e)}")