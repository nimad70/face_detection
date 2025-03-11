"""
Data Augmentation for Image Classification
"""

import random
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Path contants
DATASET_PATH = Path("dataset")
DATA_PATH = Path("data")

# Directories for train/validation/test datasets
TRAIN_DATA_PATH = DATA_PATH / "train"
# VAL_DATA_PATH = DATA_PATH / "val"
# TEST_DATA_PATH = DATA_PATH / "test"
LABELS = ["smile", "nosmile"]

# Create directories if they do not exist already
# for dirs in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]: 
#     for label in LABELS:
#         (dirs / label).mkdir(parents=True, exist_ok=True)

for label in LABELS:
        (TRAIN_DATA_PATH / label).mkdir(parents=True, exist_ok=True)


IMG_SIZE = 180

# save time, memory, and computing resources by resizing and rescaling images before training
def resize_rescale_images(image):
    """
    Resize images to the IMG_SIZE constant for shape consistent and rescale pixel values.
    """
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE), 
        layers.Rescaling(1./255)
    ])
    result = resize_and_rescale(image)
    return result


# Apply data augmentation to the input image
def data_augmentation(image):
    """
    Apply data augmentation to the input image.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"), # Randomly flip the image
        layers.RandomRotation(0.1), # Randomly rotate the image
        layers.RandomZoom(0.1), # Randomly zoom the image
        layers.RandomContrast(0.1) # Randomly adjust the contrast

    ])
    result = data_augmentation(image)
    return result


# Save augmented images to the training dataset
def save_augmented_images():
    for label in LABELS:
        image_directory = DATASET_PATH / label

        # valid_extensions = [".jpg", ".jpeg", ".png"]
        # all_images = [img for img in image_directory.iterdir() if img.suffix in valid_extensions]
        all_images = list(image_directory.iterdir())
        
        # Augment 50% of images
        half_augment = len(all_images) // 2 
        
        # Randomly select images to augment
        selected_images = random.sample(all_images, half_augment)

        for image_path in selected_images:  # Select only half of the images
            image = cv2.imread(str(image_path))
            image = resize_rescale_images(image)

            # Skip unreadable images
            if image is None:
                continue

            # Add batch dimension
            image = np.expand_dims(img, axis=0)
            augmented_image = data_augmentation(image)
            
            # Convert back to the original shape
            augmented_image = np.squeeze(augmented_image.numpy().astype("uint8"))

            # Save augmented images
            filename = f"face_{int(time.time())}.jpg"
            aug_path = TRAIN_DATA_PATH / label / filename
            cv2.imwrite(str(aug_path), augmented_image)
