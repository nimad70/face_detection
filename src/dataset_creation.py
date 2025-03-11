"""
Dataset Splitting Script

This script automates the splitting of image data into training, validation, and testing datasets. 
It is specifically designed for a binary classification task to organize data into structured directories for easy access during image processing model training.
"""

from pathlib import Path
import shutil
import random


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

def split_dataset(labels):
    """
    Splits images into training, validation, and test sets based on predefined ratios.

    Args:
        labels (list): List of labels or categories to split (e.g., ['smile', 'nosmile']).

    Returns:
        bool: True if any category has insufficient data to split; otherwise, False.
    """
    is_small_to_split = True

    for label in labels:
        # Fetch and shuffle image files from the dataset directory
        # Define the split ratios
        train_ratio = 0.6
        val_ratio = 0.2
        # test_ratio = 0.2

        image_files = list((DATASET_PATH / label).glob("*.jpg")) # Retrieved the image files 
        random.shuffle(image_files) # shuffle the images randomly
        len_all_images = len(image_files) # Get the total number of images
        print(f"Total {label} images: {len_all_images}")

        train_count = int(len(image_files) * train_ratio) # Calculate the number of training images
        val_count = int(len(image_files) * val_ratio) # Calculate the number of validation images

        # Split the images into training, validation, and test sets if there are more than 5 images
        if len_all_images > 5:
            for i, image in enumerate(image_files):
                if i < train_count:
                    destination_path = TRAIN_DATA_PATH / label / image.name
                elif i < train_count + val_count:
                    destination_path = VAL_DATA_PATH / label / image.name
                else:
                    destination_path = TEST_DATA_PATH / label / image.name

                shutil.copy(str(image), str(destination_path)) # Copy the image to the destination path
        else:
            is_small_to_split = False
            print(f"The total number of '{label}' images are too small to split, \n")
    
    return is_small_to_split