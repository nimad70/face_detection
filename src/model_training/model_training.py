"""
Model Training and fine-tuning Script for Smile Detection

This script trains a binary classification model using the MobileNetV2 architecture.
It includes data preprocessing, model building, training, and fine-tuning functionalities.

"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Path Constants
DATASET_PATH = Path("dataset")
DATA_PATH = Path("data")
MODEL_PATH = Path("model")

# Directories for train/validation/test datasets
TRAIN_DATA_PATH = DATA_PATH / "train"
VAL_DATA_PATH = DATA_PATH / "val"
TEST_DATA_PATH = DATA_PATH / "test"
LABELS = ["smile", "nosmile"]

# Create directories if they do not exist already
for dirs in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]: 
    for label in LABELS:
        (dirs / label).mkdir(parents=True, exist_ok=True)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
EPOCHS_FINE_TUNE = 10
AUTOTUNE = tf.data.AUTOTUNE # To allow tensflow to automatically tune the buffer size for optimal performance


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
    """
    Preprocesses the dataset by resizing, shuffling, and applying augmentations.

    Args:
        ds (tf.data.Dataset): Input dataset.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.

    Returns:
        tf.data.Dataset: Preprocessed dataset.
    """
    # Resize datasets
    ds = ds.map(lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    if augment:

        # Apply crop to each image of the batch
        ds = ds.map(lambda x, y: (tf.map_fn(lambda img: tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3]), x), y),num_parallel_calls=AUTOTUNE)
        
        # Randomly flip images horizontally
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE) 
        
        # Randomly adjust brightness of images
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


def load_base_model(is_fine_tuned=False):
    """
    Load the pre-trained MobileNetV2 model.

    Args:
        is_fine_tuned (bool, optional): Whether to enable fine-tuning. Defaults to False.
    
    Returns:
        base_model (tf.keras.Model): Pre-trained MobileNetV2 model.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),  # Compatible input size
        include_top=False,
        weights="imagenet"
    )

    print(f"Total number of the base model layers: {len(base_model.layers)}")

    if is_fine_tuned:
        # Enable fine-tuning
        base_model.trainable = True
    else:
        # Freeze layers
        base_model.trainable = False

    return base_model


def build_model(is_fine_tuned):
    """
    Builds the model using the base MobileNetV2 architecture.

    Args:
        is_fine_tuned (bool, optional): Whether to fine-tune the model. Defaults to False.

    Returns:
        history_initial (tf.keras.Model): The compiled model.
    """
    base_model = load_base_model(is_fine_tuned)

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # Binary classification
    ])

    return model


def train_model():
    """
    Train the model.
    
    Returns:
        history_initial (tf.keras.callbacks.History): Training history.
    """
    train_ds, val_ds, test_ds = preprocessed_data()
    model = build_model(is_fine_tuned=False)

    if model is not None:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        print("Start training the model....")

        history_initial = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )

        model.summary()

        # Save the trained model to corresponding directory
        model_save_path = MODEL_PATH / "smile_detection_model.h5"
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")

    return history_initial
  

def fine_tune_model():
    """
    ine-tunes the pre-trained model by unfreezing part of the layers.
    
    Returns:
        history_fine_tune (tf.keras.callbacks.History): Fine-tuning history.
    """
    train_ds, val_ds, test_ds = preprocessed_data()
    model = build_model(is_fine_tuned=True)

    fine_tune_at = 77 # [totale number of base_model layers (154) / 2]

    if model is not None:
        # Fine-tune only the last {fine_tune_at} layers
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        history_fine_tune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINE_TUNE
        )

        model.summary()

        # Save the fine-tuned model to corresponding directory
        model_save_path = MODEL_PATH / "smile_detection_fine_tuned_model.h5"
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")

        return history_fine_tune
