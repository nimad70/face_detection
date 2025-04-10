"""
ResNet Face Detector using OpenCV DNN module
This script uses OpenCV's DNN module to perform face detection using a ResNet-based model.
It captures video from the webcam, processes each frame to detect faces, and draws bounding boxes around detected faces.
It also provides functionality to enable or disable GPU acceleration for the model.
"""

import cv2
import numpy as np

from src.utils.config import resNetSSD_config
from src.utils.utils import update_fps

# Initialize model
def load_detector(prototxt, caffemodel, gpu_enabled):
    """
    Load the ResNet model for face detection.

    Parameters:
        prototxt (str): Path to the model's prototxt file.
        caffemodel (str): Path to the model's caffemodel file.
        gpu_enabled (bool): Flag to enable GPU acceleration.

    Returns:
        net: Loaded ResNet model.
    """
    if not prototxt.exists() or not caffemodel.exists():
        raise FileNotFoundError("[ERROR] resNet model files are missing!")

    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    if gpu_enabled:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                raise RuntimeError("\n[INFO] GPU is enabled, but no compatible GPU found!")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[INFO] GPU is enabled, using CUDA backend.")
        except Exception as e:
            print(f"\n[ERROR] Error with GPU: {e}")
            print("[INFO] Falling back to CPU.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        print("\n[INFO] GPU is disabled in config, using CPU.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    return net


def draw_rectangle(frame, faces):
    """
    Draw rectangles around detected faces in the video frame.

    Parameters:
        frame (numpy.ndarray): The video frame to process.
        faces (list): List of detected faces with bounding box coordinates.
    """
    for i, (startX, startY, endX, endY) in enumerate(faces):
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
        label = f"Face: {i+1}"
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def detect_faces(frame, net, confidence_threshold=0.5, img_size=(300, 300)):
    """
    Detect faces in the video frame using the ResNet model.

    Parameters:
        frame (numpy.ndarray): The video frame to process.
        net: Loaded ResNet model.
        confidence_threshold (float): Confidence threshold for face detection.
        img_size (tuple): Size of the input image for the model.

    Returns:
        faces (list): List of detected faces with bounding box coordinates.
    """
    # Get frame dimensions
    height, width = frame.shape[:2] 

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, img_size, (104.0, 177.0, 123.0))

    # Sets the input blob to the network and performs forward propagation to obtain detections.
    net.setInput(blob=blob)
    detections = net.forward()

    faces = []
    for i in np.arange(detections.shape[2]): # Iterate over detected objects
        update_fps(frame) # Update FPS on the frame
        confidence = detections[0, 0, i, 2]  # Confidence score
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    
    return faces


def run_face_detection():
    """
    Run the face detection on webcam stream using ResNet.
    This function captures video from the webcam, processes each frame to detect faces,
    and draws bounding boxes around detected faces.
    It also provides functionality to enable or disable GPU acceleration for the model.
    """
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size = resNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)
    cap = cv2.VideoCapture(0)

    while True:
        res, frame = cap.read()
        if not res:
            print("[ERROR] Failed to get frame")
            break
    
        faces = detect_faces(frame, net, confidence_threshold, img_size)
        draw_rectangle(frame, faces)

        cv2.imshow("MobileNetSSD Face Detection", frame)
        
        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    """
    Main function to run the face detection script.
    """
    run_face_detection()
    


