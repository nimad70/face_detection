import sys
import cv2
import time
import csv
from pathlib import Path
import shutil
import random


# AUtomate spliting the data
# Path Constants
DATASET_PATH = Path("dataset")
DATA_PATH = Path("data")

# To create train/val/test datasets
TRAIN_DATA_PATH = DATA_PATH / "train"
VAL_DATA_PATH = DATA_PATH / "val"
TEST_DATA_PATH = DATA_PATH / "test"
LABELS = ["smile", "nosmile"]


for dirs in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]: 
    for label in LABELS:
        (dirs / label).mkdir(parents=True, exist_ok=True)


# To split the dataset
def split_dataset(labels):
    is_small_to_split = False

    # To define the ratio of the dataset
    for label in labels:
        train_ratio = 0.6
        val_ratio = 0.2
        # test_ratio = 0.2

        # To get the image files with from the asked directory
        image_files = list((DATASET_PATH / label).glob("*.jpg"))
        random.shuffle(image_files) # shuffle the images randomly
        len_all_images = len(image_files)
        print(f"Total {label} images: {len_all_images}")

        train_count = int(len(image_files) * train_ratio)
        val_count = int(len(image_files) * val_ratio)
        # test_count = int(len(image_files) * test_ratio)

        if len_all_images > 5:
            for i, image in enumerate(image_files):
                if i < train_count:
                    destination_path = TRAIN_DATA_PATH / label / image.name
                elif i < train_count + val_count:
                    destination_path = VAL_DATA_PATH / label / image.name
                else:
                    destination_path = TEST_DATA_PATH / label / image.name

                shutil.copy(str(image), str(destination_path))
        else:
            is_small_to_split = True
            print(f"The total number of '{label}' images are too small to split, \n")
    
    return is_small_to_split