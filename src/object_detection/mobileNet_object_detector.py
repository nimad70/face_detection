"""
SSD Face Detection using MobileNetSSD model.
This script captures video frames from a webcam, detects faces using the MobileNetSSD model, and displays the results in real-time.
It also supports GPU acceleration for faster processing.
"""

import cv2
import numpy as np
# from utils import update_fps
# from config import mobileNetSSD_config
from src.utils.utils import update_fps
from src.utils.config import mobileNetSSD_config

# Initialize model
def load_detector(prototxt, caffemodel, gpu_enabled):
    """
    Load the MobileNetSSD model for face detection.

    Parameters:
        prototxt (str): Path to the Caffe model prototxt file.
        caffemodel (str): Path to the Caffe model weights file.
        gpu_enabled (bool): Flag to enable GPU acceleration.

    Returns:
        net: Loaded MobileNetSSD model.
    """
    if not prototxt.exists() or not caffemodel.exists():
        raise FileNotFoundError("MobileNetSSD model files are missing in the 'model/MobileNetSSD/' directory.")

    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    if gpu_enabled:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                raise RuntimeError("GPU is enabled, but no compatible GPU found")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("GPU is enabled, using CUDA backend")
        except Exception as e:
            print(f"Error enabling GPU: {e}")
            print("Falling back to CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        print("GPU is disabled in config, using CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    return net


def draw_rectangle(frame, objects, object_classes, colors, is_object_detection=False):
    """
    Draws bounding boxes (rectangles) around detected objects on the original frame.
    
    Parameters:
        frame (numpy.ndarray): The original video frame.
        objects (list): List of detected object bounding boxes (x, y, w, h).
        object_classes (list): List of object classes.
        colors (list): List of colors for each class.
        is_object_detection (bool): Flag to indicate if it's object detection or not.
    """
    if not is_object_detection:
        for ((startX, startY, endX, endY), confidence, class_id) in objects:
            if class_id == 15: # Person class ID
                label = f"{object_classes[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        for ((startX, startY, endX, endY), confidence, class_id) in objects:
            label = f"{object_classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[class_id], 1)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 1)


def detect_objects(frame, net, confidence_threshold=0.5, img_size=(300, 300)):
    """
    Detect objects in a video frame using the MobileNetSSD model.
    
    Parameters:
        frame (numpy.ndarray): The video frame to process.
        net: Loaded MobileNetSSD model.
        confidence_threshold (float): Minimum confidence threshold for detection.
        img_size (tuple): Size of the input image for the model.
    
    Returns:
        objects (list): List of detected objects with bounding box coordinates.
    """
    # Get frame dimensions
    height, width = frame.shape[:2] 

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, img_size, 127.5)

    # Sets the input blob to the network and performs forward propagation to obtain detections.
    net.setInput(blob=blob)
    detections = net.forward()

    objects = []
    # # Only detect people
    # for i in np.arange(0, detections.shape[2]): # Iterate over detected objects
    #     update_fps(frame) # Update FPS on the frame
    #     confidence = detections[0, 0, i, 2]  # Confidence score
    #     class_id = int(detections[0, 0, i, 1]) # Class ID of detected object
        
    #     if confidence > confidence_threshold and class_id == 15:
    #         box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
    #         (startX, startY, endX, endY) = box.astype("int")
    #         objects.append(((startX, startY, endX, endY), confidence))

    for i in np.arange(0, detections.shape[2]): # Iterate over detected objects
        update_fps(frame) # Update FPS on the frame
        confidence = detections[0, 0, i, 2]  # Confidence score
        
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1]) # Class ID of detected object
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            objects.append(((startX, startY, endX, endY), confidence, class_id))

    return objects
        

def run_object_detection(is_object_detection=False):
    """
    Run the object detection using the MobileNetSSD model.

    Parameters:
        is_object_detection (bool): Flag to indicate if it's object detection or not.
    """
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size, object_classes, colors = mobileNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)
    cap = cv2.VideoCapture(0)

    while True:
        res, frame = cap.read()
        if not res:
            print("Failed to get frame")
            break
    
        objects = detect_objects(frame, net, confidence_threshold, img_size)
        draw_rectangle(frame, objects, object_classes, colors, is_object_detection)

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
    is_object_detection = True # Set to True for object detection, False for face detection
    run_object_detection(is_object_detection=is_object_detection)
    


