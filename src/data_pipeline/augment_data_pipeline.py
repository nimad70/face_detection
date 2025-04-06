"""
Data Augmentation for Image Classification

This script performs resizing, rescaling, and data augmentation on image datasets. 
Augmented images are saved to enrich the training dataset, enhancing model performance and robustness.
"""

import time
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from src.object_detection.resNet_face_detector import load_detector, detect_faces
from src.utils.config import resNetSSD_config


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


def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Applies color jittering to the input image.
    Randomly changes brightness, contrast, saturation, and hue.
    
    parameters:
        image (Tensor): Input image tensor.
        brightness (float): Brightness adjustment factor.
        contrast (float): Contrast adjustment factor.
        saturation (float): Saturation adjustment factor.
        hue (float): Hue adjustment factor.
    
    returns:
        Tensor: Jittered image tensor.
    """
    image = tf.image.random_brightness(image, max_delta=brightness)
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    image = tf.image.random_hue(image, max_delta=hue)
    return tf.clip_by_value(image, 0.0, 255.0)


def random_grayscale(image, probability=0.1):
    """
    Randomly converts an RGB image to grayscale with a given probability.

    The grayscale image is converted back to 3 channels to maintain shape consistency.
    Useful for simulating colorless environments or lighting conditions.

    Parameters
        image (tf.Tensor): A 3D tensor representing an RGB image of shape (height, width, 3).
        prob (float, optional): Probability of converting the image to grayscale (default is 0.1).

    Returns
        tf.Tensor: The original image or a grayscale version, maintaining the same shape.
    """
    if tf.random.uniform([]) < probability:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image) # convert back to 3 channels

    return image


def add_gaussian_noise(image, mean=0.0, stddev=1.0, probability=0.3):
    """
    Adds Gaussian noise to an image with a given probability.

    This augmentation simulates sensor noise and improves robustness to low-quality inputs.

    Parameters
        image (tf.Tensor): A 3D tensor representing an image of shape (height, width, channels).
        mean (float, optional): Mean of the Gaussian noise distribution (default is 0.0).
        stddev (float, optional): Standard deviation of the Gaussian noise (default is 10.0).
        prob (float, optional): Probability of applying Gaussian noise (default is 0.3).

    Returns
        tf.Tensor: The image with noise added if applied, clipped to [0.0, 255.0].
    """
    if tf.random.uniform([]) < probability:
        noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev)
        image = tf.cast(image, tf.float32) + noise
    return tf.clip_by_value(image, 0.0, 255.0)


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
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1),

    ])

    # Resize before augmentation
    resized = tf.convert_to_tensor(image, dtype=tf.float32)
    resized = tf.expand_dims(resized, axis=0)
    resized = tf.image.resize(resized, [IMG_SIZE, IMG_SIZE])

    augmented = data_augmentation(resized)[0]
    augmented = color_jitter(augmented)
    augmented = random_grayscale(augmented)
    augmented = add_gaussian_noise(augmented)

    # augmented = np.squeeze(augmented.numpy().astype("uint8"))
    
    return augmented.numpy().astype("uint8")


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
            count_augmented = 0
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
                
                # Apply 5 augmentations per image
                for i in range(5):
                    augmented = augment_image(image)

                # Save augmented image
                filename = f"face_{int(time.time()*1000)}.jpg"
                aug_path = train_path / filename
                cv2.imwrite(str(aug_path), augmented)
                count_augmented += 1
    
            is_augmented = True
            print("\n[INFO] Data augmentation process complete.")
            print(f"[INFO] {count_augmented} images augmented and saved to {train_path}.")
        
        except Exception as e:
            print(f"[ERROR] Augmentation failed: {e}")
    
    return is_augmented

