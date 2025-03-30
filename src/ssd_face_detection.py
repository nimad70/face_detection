import cv2
import numpy as np
import os


# Load OpenCV deep learning-based model (SSD face detector)
MODEL_BASE_DIR = "model\MobileNetSSD"
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "\MobileNetSSD_deploy.prototxt"
GPU_SUPPORT = 0
IMG_SIZE = (300, 300)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "face"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Load the pre-trained model
prototxt = os.path.join(MODEL_BASE_DIR, PROTOTXT)
model = os.join.path(MODEL_BASE_DIR, MODEL)

if not os.path.exists(prototxt) and os.path.exists(model):
    raise FileNotFoundError("MobileNetSSD model files are missing. Ensure they are in the 'models/' directory.")

net = cv2.dnn.readNetFromCaffe(prototxt, model)

if GPU_SUPPORT:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)

while True:
    res, frame = cap.read()
    if not res:
        print("Failed to get frame")
        break

    # Get frame dimensions
    height, width = frame.shape[:2] 

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, IMG_SIZE, 127.5)

    # Sets the input blob to the network and performs forward propagation to obtain detections.
    # 
    net.setInput(blob=blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]): # Iterate over detected objects
        confidence = detections[0, 0, i, 2]  # Confidence score
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1]) # Class ID of detected object
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")





