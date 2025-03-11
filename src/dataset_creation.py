import sys
import cv2
import time
import csv
from pathlib import Path
import shutil
import random



# To automate the process of creating the dataset for smile and no smile images
# Path Constants
DATASET_PATH = Path("dataset")
SMILE_PATH = DATASET_PATH / "smile"
NO_SMILE_PATH = DATASET_PATH / "nosmile"

# To create directory if they don't exist already, create missing parents if needed
SMILE_PATH.mkdir(parents=True, exist_ok=True)
NO_SMILE_PATH.mkdir(parents=True, exist_ok=True)


# AUtomate spliting the data
# To create train/val/test datasets
DATA_PATH = Path("data")
TRAIN_DATA_PATH = DATA_PATH / "train"
VAL_DATA_PATH = DATA_PATH / "val"
TEST_DATA_PATH = DATA_PATH / "test"
LABELS = ["smile", "nosmile"]


for dirs in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]: 
    for label in LABELS:
        (dirs / label).mkdir(parents=True, exist_ok=True)

# https://docs.python.org/3/library/shutil.html
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


def creat_dir(frame, faces):
    is_proceed = False
    # To save detected faces
    with open(DATASET_PATH / "labels.csv", mode="a", newline="") as csvfile:
        label_writer = csv.writer(csvfile)
        selected_key = cv2.waitKey(1) & 0xFF

        if selected_key == ord('s') or selected_key == ord('a'):
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                filename = f"face_{int(time.time())}.jpg"
                label = "smile" if selected_key == ord('s') else 'nosmile'
                path = SMILE_PATH / filename if selected_key == ord('s') else NO_SMILE_PATH /filename
                cv2.imwrite(str(path), face)

                # To write the label in a CSV file
                label_writer.writerow([filename, label])

        if selected_key == ord('q'):
            return is_proceed
