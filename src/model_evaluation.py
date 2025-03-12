import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Constants
DATA_PATH = Path("data")
IMG_SIZE = 224
BATCH_SIZE = 32

# Load test dataset
TEST_DATA_PATH = Path("data/test")


def load_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DATA_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_names = test_ds.class_names
    print(f"Test dataset class names: {class_names}")
    return test_ds
