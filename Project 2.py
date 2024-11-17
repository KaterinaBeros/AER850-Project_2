# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:37:07 2024

@author: beros
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
import tensorflow as tf
from keras import layers
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# 2.1 Step 1: Data Processing 
#Define parameters for image processing
IMAGE_WIDTH = 500  #Width of input images 
IMAGE_HEIGHT = 500  #Height of input images
CHANNELS = 3        #Number of colour channels
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)


#Define paths
train_path=r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\Data\test"
val_path=r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\Data\valid"
test_path=r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\Data\train"

print('Importing Data...')

# Create data generator with augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,             #Normalize pixel values
    rotation_range=40,          #Apply rotation
    shear_range=0.2,            #Apply shear transformation
    zoom_range=0.2,             #Apply random zoom
    horizontal_flip=True,       #Enable horizontal flipping
    vertical_flip=True          #Enable vertical flip
)

# Create data generator for validation (only rescaling)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

# Set up generator for training data
train_generator = train_datagen.flow_from_directory(
    train_path, #Training directory
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=32,
    class_mode='categorical', #Used for multi-class classification 
    color_mode='rgb', #For 3 channels
    shuffle=True        #Shuffle training data 
)

#Set up generator for validation data
validation_generator = validation_datagen.flow_from_directory(
    val_path,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)
# Create test data generator (for later use)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False  # Don't shuffle test data
)

# Print information about the dataset
print("\nDataset Information:")
print("Training set:", train_generator.samples, "images")
print("Validation set:", validation_generator.samples, "images")
print("Test set:", test_generator.samples, "images")
print("\nClass mapping:", train_generator.class_indices)

#2.2 Step 2: Neural Network Architecture Design & 2.4 Hyperparameter Analysis
model = Sequential([
    #First block: Inital feature detection
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(500,500,3)),
    layers.MaxPooling2D(),
    
    #Second block: More sepcific feature learning
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5), #Prevents overfitting
    
    #Third block: Complex feature detection
    layers.Conv2D(128,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    
    #Classification Layers
    layers.Flatten(), #Convert features to 1D
    layers.Dense(144,activation='relu'), #Combine features 
    layers.Dropout(0.5), #Final regularization
    layers.Dense(3, activation='softmax') #Output probabilities for 3 classes 
    ])

# Configure model with standard classification settings
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


#2.4 Step 4 : Model Evaluation 
#Training model with 60 epochs ***
m=model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs=30)

#Display model architecture
model.summary()

#Extract training history
acc = m.history['accuracy']
val_acc = m.history['val_accuracy']
loss = m.history['loss']
val_loss = m.history['val_loss']

# Creating better visualization with a side-by-side comparison
plt.figure(figsize=(12, 4))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()

# Print final metrics for numerical analysis
print(f"\nFinal Training Accuracy: {acc[-1]:.4f}")
print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
print(f"Final Training Loss: {loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")

#Saving Model
model_save_path = os.path.join(r"C:\Users\beros\OneDrive\Documents\AER 850 Machine Learning\Project 2\AER850_Project2.h5")
model.save(model_save_path)
print("Model saved as AER850_Project2.h5")
