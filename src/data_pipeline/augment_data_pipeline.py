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
from src.utils.config import resNetSSD_config
from src.object_detection.resNet_face_detector import load_detector, detect_faces


# Path contants
DATASET_PATH = Path("dataset")
DATA_PATH = Path("data")
TRAIN_DATA_PATH = DATA_PATH / "train"
LABELS = ["smile", "nosmile"]

# Create directories for corresponding training datasets if does not exist
for label in LABELS:
    (TRAIN_DATA_PATH / label).mkdir(parents=True, exist_ok=True)

IMG_SIZE = 180


def crop_face(image):
    """
    Detect and return the cropped face in the image.
    
    Parameters:
        image (np.ndarray): Input image.
    Returns:    
        cropped_face (np.ndarray): Cropped face image if detected, otherwise None.
    """
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size = resNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)
    faces = detect_faces(image, net)
    if len(faces) == 0:
        return None
    
    if faces:
        (startX, startY, endX,endY) = faces[0]
        cropped_face = image[startY:endY, startX:endX]
        
        if cropped_face.size != 0:
            return cropped_face
    
    return None


"""
Rescalling is better to happen after augmentation, and during model training,
not before saving to disk
"""
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


def augment_image(image):
    """
    Applies data augmentation to an input image, including random flipping, rotation, zoom, and contrast adjustment.

    The image is first resized to [IMG_SIZE, IMG_SIZE] before applying augmentations.

    Parameters:
        image (Tensor): Input image tensor. Expected to be a 3D tensor representing a single image.

    Returns:
        Tensor: Augmented image tensor with the same shape as the resized input (IMG_SIZE, IMG_SIZE, channels).
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)

    ])

    # Resize before augmentation
    resized = tf.convert_to_tensor(image, dtype=tf.float32)
    resized = tf.expand_dims(resized, axis=0)
    resized = tf.image.resize(resized, [IMG_SIZE, IMG_SIZE])

    augmented = data_augmentation(resized)
    augmented = np.squeeze(augmented.numpy().astype("uint8"))

    # augmented = image_augmentation(resized)
    return augmented


def save_augmented_images():
    """
    Augments (e.x. 80%) of the dataset by detecting and cropping faces before augmentation.
    Saves new images to the train directory.
    """
    is_augmented = False

    for label in LABELS:
        image_directory = DATASET_PATH / label
        if not image_directory.exists():
            print(f"[WARNING] Directory {image_directory} does not exist. Skipping.")
            continue
        
        valid_extensions = [".jpg", ".jpeg", ".png"]
        all_images = [img for img in image_directory.iterdir() if img.suffix in valid_extensions]
        # all_images = list(image_directory.iterdir())
        
        # Randomly select images to augment
        augment_size = int(len(all_images) * 0.8)
        selected_images = random.sample(all_images, augment_size)

        try:
            train_path = TRAIN_DATA_PATH / label
            if not train_path.exists():
                print(f"[WARNING] Directory {train_path} does not exist. Skipping.")
                continue

            for image_path in selected_images:
                # print(f"[INFO] Processing: {image_path.name}")
                image = cv2.imread(str(image_path))
                if image is None:
                    print("[WARNING] Failed to load image. Skipping.")
                    continue
                
                # Apply augmentation and convert back to the original shape
                augmented = augment_image(image)

                # Save augmented image
                filename = f"face_{int(time.time()*1000)}.jpg"
                aug_path = train_path / filename
                cv2.imwrite(str(aug_path), augmented)
                is_augmented = True
                print("\n [INFO] Data augmentation process complete.")
                print("[INFO] Augmented images are saved to the corresponding training datasets!")
        except Exception as e:
            print(f"[ERROR] Augmentation failed: {e}")
    
    return is_augmented

