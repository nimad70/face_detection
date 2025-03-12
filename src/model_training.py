"""
Model training script
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path


# Path Constants
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

IMG_SIZE = 256
BATCH_SIZE = 32

# To allow tensflow to automatically tune the buffer size for optimal performance
AUTOTUNE = tf.data.AUTOTUNE

# Loading data from the dataset directory
def load_data():
    """
    Load the dataset from the dataset directory.
    
    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        test_ds (tf.data.Dataset): Testing dataset.
    """
    train_ds = keras.preprocessing.image_dataset_from_directory(
        str(TRAIN_DATA_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        str(VAL_DATA_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    test_ds = keras.preprocessing.image_dataset_from_directory(
        str(TEST_DATA_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Class Names:", class_names)

    return train_ds, val_ds, test_ds


# Apply data pre-processing to process train, valid, and test sets
def prepare(ds, shuffle=False, augment=False):
    # Resize datasets
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    if augment:
        # Randomly flip images horizontally
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE) 
        
        # Randomly rotate images
        ds = ds.map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.3), y), num_parallel_calls=AUTOTUNE)
        
        # Randomly adjust the contrast of images
        ds = ds.map(lambda x, y: (tf.image.random_contrast(x, lower=0.5, upper=1.3), y), num_parallel_calls=AUTOTUNE)
        
        # Randomly adjust the hue of images
        ds = ds.map(lambda x, y: (tf.image.random_hue(x, max_delta=0.05), y), num_parallel_calls=AUTOTUNE)

        # Randomly adjust the saturation of images
        ds = ds.map(lambda x, y: (tf.image.random_saturation(x, lower=0.5, upper=3), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching to ensures the next batch is prepared while the current batch is being processed
    return ds.prefetch(buffer_size=AUTOTUNE)


def preprocessed_data():
    """
    Load the dataset and apply pre-processing.
    
    Returns:
        train_ds (tf.data.Dataset): Preprocessed training dataset.
        val_ds (tf.data.Dataset): Preprocessed validation dataset.
        test_ds (tf.data.Dataset): Preprocessed testing dataset.
    """
    train_ds, val_ds, test_ds = load_data()
    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)
    test_ds = prepare(test_ds)

    return train_ds, val_ds, test_ds
