"""
Emotion Detection using EfficientNetB0 and ResNet SSD.
This script performs emotion detection using a TFLite model based on EfficientNetB0 architecture.
"""

import time

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils.config import resNetSSD_config


# Constants
IMG_SIZE = 224
TFLITE_MODEL_PATH = Path("model/EfficientNetB0/smile_detection_model.tflite")


def run_emotion_detection(image, confidence_threshold=0.5, show_boxes=True):
    """
    Run emotion detection on the input image using a TFLite model.
    
    Returns:
        image with detected faces and their corresponding emotion labels.
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load ResNet SSD face detector
    gpu_enabled, prototxt, caffemodel, _, input_size = resNetSSD_config()
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Process the image
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, input_size, (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(face_normalized, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            label = f"Smiling ({prediction:.2f})" if prediction > 0.5 else f"Not Smiling ({prediction:.2f})"
            color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)

            if show_boxes:
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image
