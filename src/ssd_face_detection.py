import cv2
import numpy as np


# Load OpenCV deep learning-based model (SSD face detector)
model_path = "opencv_face_detector_unit8.pb"
config_path = "opencv_face_detector.pbtxt"

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow(model=model_path, config=config_path)