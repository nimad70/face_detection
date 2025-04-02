"""
Data Augmentation for Image Classification

This script performs resizing, rescaling, and data augmentation on image datasets. 
Augmented images are saved to enrich the training dataset, enhancing model performance and robustness.
"""

import random
import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Path contants
DATASET_PATH = Path("dataset")
DATA_PATH = Path("data")

# Directories for training datasets
TRAIN_DATA_PATH = DATA_PATH / "train"
LABELS = ["smile", "nosmile"]

# Create directories for corresponding training datasets if does not exist
for label in LABELS:
        (TRAIN_DATA_PATH / label).mkdir(parents=True, exist_ok=True)

IMG_SIZE = 180


def resize_images(image):
    """
    Resizes input images to IMG_SIZE for consistent dimensions to save time, memory, and computing resources

    Args:
        image (np.ndarray): Input image tensor.

    Returns:
        result (Tensor): Resized image tensor.
    """
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE), 
    ])
    result = resize_and_rescale(image)
    return result


def resize_rescale_images(image):
    """
    Resizes and rescales input images to IMG_SIZE and normalizes pixel values.

    Args:
        image (np.ndarray): Input image tensor.

    Returns:
        result (Tensor): Resized and normalized image tensor.
    """
    image = tf.cast(image, tf.float32)
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE), 
        layers.Rescaling(1./255)
    ])
    result = resize_and_rescale(image)
    
    return result


def data_augmentation(image):
    """
    Applies data augmentation techniques such as random flipping, rotation, zoom, and contrast adjustment.

    Args:
        image (Tensor): Input image tensor.

    Returns:
        result (Tensor): Augmented image tensor.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"), # Randomly flip the image
        layers.RandomRotation(0.1), # Randomly rotate the image
        layers.RandomZoom(0.1), # Randomly zoom the image
        layers.RandomContrast(0.1) # Randomly adjust the contrast

    ])
    result = data_augmentation(image)
    return result


def save_augmented_images():
    """
    Augments and saves 50% of images from the dataset to the corresponding training directory.

    Reads images, applies resizing and augmentation, then saves the augmented images.
    """
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
            image = resize_images(image)

            # Skip unreadable images
            if image is None:
                continue

            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            augmented_image = data_augmentation(image)
            
            # Convert back to the original shape
            augmented_image = np.squeeze(augmented_image.numpy().astype("uint8"))

            # Save augmented images
            filename = f"face_{int(time.time())}.jpg"
            aug_path = TRAIN_DATA_PATH / label / filename
            cv2.imwrite(str(aug_path), augmented_image)
