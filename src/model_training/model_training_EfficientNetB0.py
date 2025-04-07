"""
Model Training and fine-tuning Script for Smile Detection

This script trains a binary classification model using the ----- architecture.
It includes data preprocessing, model building, training, and fine-tuning functionalities.

"""
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tqdm import tqdm 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


# --- Configuration via input ---
def get_user_input():
    use_split = use_split = input("Use pre-split dataset with train/val/test directories? (y/n): ").strip().lower() == 'y'
    is_augmented = input("Has data augmentation already been applied to training data? (y/n): ").strip().lower() == 'y'
    return use_split, is_augmented

USE_PRE_SPLIT, AUGMENTED = get_user_input() # True = use train/val/test folders; False = use raw smile/nosmile

# --- Constants ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
EPOCHS_FINE_TUNE = 10
AUTOTUNE = tf.data.AUTOTUNE # To allow tensflow to automatically tune the buffer size for optimal performance

# --- Paths ---
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

MODEL_PATH.mkdir(parents=True, exist_ok=True)


# --- Utility Functions ---
def load_and_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_input(image)
    return image


def prepare(ds, shuffle=False, augment=False):
    """
    Preprocesses the dataset by resizing, shuffling, and applying augmentations.

    Args:
        ds (tf.data.Dataset): Input dataset.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.

    Returns:
        tf.data.Dataset: Preprocessed dataset. Used buffered prefetching to ensures the next batch is prepared while the current batch is being processed.
    """
    # Resize datasets
    # ds = ds.map(lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    if augment:
        ds = ds.map(lambda x, y: (tf.map_fn(lambda img: tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3]), x), y),num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE) 
        ds = ds.map(lambda x, y: (tf.image.random_brightness(x, max_delta=0.3), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_contrast(x, lower=0.5, upper=1.3), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_hue(x, max_delta=0.05), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_saturation(x, lower=0.5, upper=3), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


#  --- Data Loading ---
def load_data_from_split():
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


def load_data_from_raw():
    """
    Load the dataset from the raw dataset directory and split it into train, validation, and test sets.
    
    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        test_ds (tf.data.Dataset): Testing dataset.
    """
    all_image_paths = []
    all_labels = []
    for label in LABELS:
        label_dir = DATASET_PATH / label
        image_paths = list(label_dir.glob("*.jpg"))
        all_image_paths.extend(image_paths)
        all_labels.extend([label] * len(image_paths))

    label_to_index = {name: idx for idx, name in enumerate(LABELS)}
    all_labels = [label_to_index[label] for label in all_labels]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.25, stratify=train_labels, random_state=42)  # 0.25 x 0.8 = 0.2

    def path_to_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), labels))
        ds = ds.map(lambda x, y: (load_and_preprocess(x), y), num_parallel_calls=AUTOTUNE)
        return ds

    return (
        path_to_dataset(train_paths, train_labels).batch(BATCH_SIZE),
        path_to_dataset(val_paths, val_labels).batch(BATCH_SIZE),
        path_to_dataset(test_paths, test_labels).batch(BATCH_SIZE)
    )


def preprocessed_data(apply_augmentation):
    """
    Load the dataset and apply pre-processing.
    
    Returns:
        train_ds (tf.data.Dataset): Preprocessed training dataset.
        val_ds (tf.data.Dataset): Preprocessed validation dataset.
        test_ds (tf.data.Dataset): Preprocessed testing dataset.
    """
    if USE_PRE_SPLIT:
        train_ds, val_ds, test_ds = load_data_from_split()
    else:
        train_ds, val_ds, test_ds = load_data_from_raw()

    train_ds = prepare(train_ds, shuffle=True, augment=apply_augmentation)
    val_ds = prepare(val_ds)
    test_ds = prepare(test_ds)

    return train_ds, val_ds, test_ds


# --- Model Architecture ---
def load_base_model(is_fine_tuned=False):
    """
    Load the pre-trained MobileNetV2 model.

    Args:
        is_fine_tuned (bool, optional): Whether to enable fine-tuning. Defaults to False.
    
    Returns:
        base_model (tf.keras.Model): Pre-trained MobileNetV2 model.
    """
    base_model = keras.applications.EfficientNetV2B0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = is_fine_tuned
    print(f"Total number of the base model layers: {len(base_model.layers)}")

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
        layers.Dense(1, activation="sigmoid")
    ])

    return model


# --- Hyperparameter Tuning Support ---
def build_model_with_hyperparam_tuning(lr=0.0001, dense=256, dropout=0.5, freeze=True):
    base_model = keras.applications.EfficientNetV2B0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = not freeze

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid")
    ])

    print(f"[HYPERPARAM LOG] lr: {lr}, dense: {dense}, dropout: {dropout}, freeze: {freeze}")
    return model


# --- Plot and Save Training History ---
def save_history_plot(history, tag):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()

    plot_path = MODEL_PATH / f"training_history_{tag}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training plot: {plot_path}")


# --- K-Fold Training (if raw dataset used) ---
# --- K-Fold Training ---
def train_with_kfold(augmented):
    all_image_paths, all_labels = [], []
    for label in LABELS:
        image_paths = list((DATASET_PATH / label).glob("*.jpg"))
        all_image_paths.extend(image_paths)
        all_labels.extend([label] * len(image_paths))

    label_to_index = {name: idx for idx, name in enumerate(LABELS)}
    all_labels = [label_to_index[label] for label in all_labels]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies, fold_params = [], []

    for fold, (train_index, val_index) in enumerate(skf.split(all_image_paths, all_labels)):
        print(f"\n--- Fold {fold + 1} ---")
        train_paths = [all_image_paths[i] for i in train_index]
        val_paths = [all_image_paths[i] for i in val_index]
        train_labels = [all_labels[i] for i in train_index]
        val_labels = [all_labels[i] for i in val_index]

        def path_to_ds(paths, labels):
            ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), labels))
            ds = ds.map(lambda x, y: (load_and_preprocess(x), y), num_parallel_calls=AUTOTUNE)
            return ds

        train_ds = prepare(path_to_ds(train_paths, train_labels).batch(BATCH_SIZE), shuffle=True, augment=not augmented)
        val_ds = prepare(path_to_ds(val_paths, val_labels).batch(BATCH_SIZE))

        best_model, best_score, best_params = None, 0, None
        param_grid = {
            'lr': [0.0001, 0.0005, 0.001],
            'dense': [128, 256, 512],
            'dropout': [0.3, 0.5],
            'freeze': [True, False]
        }
        search_space = list(ParameterSampler(param_grid, n_iter=5, random_state=fold))

        for i, params in enumerate(search_space):
            print(f"[Tuning] Trying config {i+1}/{len(search_space)}: {params}")
            model = build_model_with_hyperparam_tuning(**params)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                          loss="binary_crossentropy",
                          metrics=["accuracy"])
            hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)
            val_acc = hist.history['val_accuracy'][-1]
            if val_acc > best_score:
                best_model, best_score, best_params, best_history = model, val_acc, params, hist

        print(f"[INFO] Fold {fold+1} Best Hyperparameters: {best_params}")

        y_true, y_pred = [], []
        for batch in val_ds:
            images, labels = batch
            preds = best_model.predict(images)
            preds = np.round(preds).astype(int).flatten()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred)
        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
        fold_params.append(best_params)
        accuracies.append(acc)
        model_filename = f"smile_model_fold{fold + 1}_acc_{acc:.4f}.h5"
        best_model.save(MODEL_PATH / model_filename)
        save_history_plot(best_history, f"kfold_fold{fold + 1}")

    pd.DataFrame({"accuracy": accuracies, "params": fold_params}).to_csv(MODEL_PATH / "results_kfold.csv", index=False)
    print("Saved fold results to results_kfold.csv")



# --- Pre-Split Training ---
def fine_tune_layer_input():
    base_model_ = load_base_model()
    fine_tune_at = int(input("Enter the number of layers to fine-tune (from the end): "))
    return fine_tune_at


def train_model(augmented=False):
    train_ds, val_ds, test_ds = preprocessed_data(apply_augmentation=not augmented)
    best_model, best_score, best_params = None, 0, None
    param_grid = {
        'lr': [0.0001, 0.0005, 0.001],
        'dense': [128, 256, 512],
        'dropout': [0.3, 0.5],
        'freeze': [True, False]
    }
    search_space = list(ParameterSampler(param_grid, n_iter=5, random_state=42))

    for i, params in enumerate(search_space):
        print(f"[PreSplit Tuning] Trying config {i+1}/{len(search_space)}: {params}")
        model = build_model_with_hyperparam_tuning(**params)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)
        val_acc = hist.history['val_accuracy'][-1]
        if val_acc > best_score:
            best_model, best_score, best_params, best_history = model, val_acc, params, hist

    print(f"[INFO] Best Hyperparameters (PreSplit): {best_params}")
    save_history_plot(best_history, "presplit")
    best_model.save(MODEL_PATH / "smile_detection_model.h5")
    print("Model saved at smile_detection_model.h5")

    with open(MODEL_PATH / "best_params_presplit.txt", "w") as f:
        f.write(str(best_params))
        f.write(f"\nBest validation accuracy: {best_score:.4f}")

    return best_history
  

def fine_tune_model():
    """
    ine-tunes the pre-trained model by unfreezing part of the layers.
    
    Returns:
        history_fine_tune (tf.keras.callbacks.History): Fine-tuning history.
    """
    train_ds, val_ds, test_ds = preprocessed_data(apply_augmentation=False)
    model = build_model(is_fine_tuned=True)

    fine_tune_at = fine_tune_layer_input()

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


if __name__ == "__main__":
    if USE_PRE_SPLIT:
        history = train_model(augmented=AUGMENTED)
        save_history_plot(history, "presplit")
        fine_tune_model()
    else:
        train_with_kfold(augmented=AUGMENTED)