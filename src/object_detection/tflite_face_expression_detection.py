"""
Real-time smile detection using TFLite model and ResNet SSD face detector.
This script captures video from the webcam, detects faces using a pre-trained ResNet SSD model.
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils.config import resNetSSD_config


# Constants
IMG_SIZE = 224
TFLITE_MODEL_PATH = Path("model/EfficientNetB0/smile_detection_model.tflite")


def face_expression_detection_tflite():
    """
    Real-time smile detection using TFLite model and ResNet SSD face detector.
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load ResNet SSD face detector
    gpu_enabled, prototxt, caffemodel, confidence_threshold, input_size = resNetSSD_config()
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    if gpu_enabled and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, input_size, (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face_normalized = face_resized.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(face_normalized, axis=0)

                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

                label = "Smiling" if prediction > 0.5 else "Not Smiling"
                color = (0, 255, 0) if label == "Smiling" else (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Real-Time Smile Detection (TFLite + ResNet)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
