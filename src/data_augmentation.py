"""
Data Augmentation for Image Classification
"""

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
VAL_DATA_PATH = DATA_PATH / "val"
TEST_DATA_PATH = DATA_PATH / "test"
LABELS = ["smile", "nosmile"]

# Create directories if they do not exist already
for dirs in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]: 
    for label in LABELS:
        (dirs / label).mkdir(parents=True, exist_ok=True)


IMG_SIZE = 180

def resize_images(image):
    """
    Resize images to the IMG_SIZE constant for shape consistent and rescale pixel values.
    """
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE), 
        layers.Rescaling(1./255)
    ])
    result = resize_and_rescale(image)
    return result


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






