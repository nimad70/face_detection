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


# Loading data
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
