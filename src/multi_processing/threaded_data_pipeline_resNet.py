"""
Threaded face detection using ResNet model.
This script captures video frames from a webcam, detects faces using a ResNet model, and displays the results in real-time.
It uses threading to handle the detection process separately from the video capture, allowing for smoother performance.
"""

import csv
import time
from pathlib import Path

import cv2

from src.multi_processing.threading_detection_handler import DetectionThread
from src.multi_processing.video_stearm import VideoStream
from src.object_detection.resNet_face_detector import (
    load_detector, 
    detect_faces, 
    draw_rectangle,
)
from src.utils.config import resNetSSD_config


# Define the path to save the dataset
DATASET_PATH = Path("dataset")
SMILE_PATH = DATASET_PATH / "smile"
NO_SMILE_PATH = DATASET_PATH / "nosmile"

# Create directory if they do not exist already
SMILE_PATH.mkdir(parents=True, exist_ok=True)
NO_SMILE_PATH.mkdir(parents=True, exist_ok=True)

# Shared variables
current_frame = None
detected_faces = []
processed_frames= []

def get_frame():
    """
    Get the current frame from the video stream.
    """
    return current_frame


def set_result(frame, faces):
    """
    Set the detection results.
    """
    global detected_faces
    detected_faces = faces


def display_menu():
    """
    Displays user interaction options.
    """
    print("\nOptions: \n\n"
    "  => Press 's' = Save smiling face\n"
    "  => Press 'a' = save none-smiling face\n"
    "  => Press 'q' = Quit\n")


def save_face_cross(frame, faces, label, label_writer):
    """
    Save the detected face with the specified label.
    """
    target_path = SMILE_PATH if label == 'smile' else NO_SMILE_PATH
    for (startX, startY, endX, endY) in faces:
        # Safe bounding box cropping
        face = frame[max(0, startY):endY, max(0, startX):endX]
        if face.size == 0:
            continue  # Skip if the face is empty
        
        filename = f"face_{int(time.time()*1000)}.jpg"
        file_path = target_path / filename
        cv2.imwrite(str(file_path), face)  # Save the face image
        label_writer.writerow([filename, label])  # Write the label in a CSV file


def data_pipeline_thread():
    """
    Main function to run the threaded face detection using ResNet model.

    returns:
        bool: True if the process was successful, False otherwise.
    """
    global current_frame

    vs = VideoStream().start()
    
    # Load the detector
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size = resNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)

    # Wrap the detection function
    detector = lambda frame: detect_faces(frame, net)
    
    # Create and start the detection thread
    detection_thread = DetectionThread(detector, get_frame, set_result)
    detection_thread.start()
    

    try:
        with open(DATASET_PATH / "labels.csv", mode="a", newline="") as csvfile:
            label_writer = csv.writer(csvfile) # Create a CSV writer object
            
            display_menu()
            while True:
                ret, frame = vs.read()
                if not ret:
                    break

                current_frame = frame.copy()
                draw_rectangle(current_frame, detected_faces)
                cv2.imshow("Threaded Face Detection", current_frame)

                # To save the detected faces inside smile/nosmile directories
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    save_face_cross(frame, detected_faces, "smile", label_writer)
                elif key == ord('a'):
                    save_face_cross(frame, detected_faces, "nosmile", label_writer)
                elif key == ord('q'):
                    break
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return False

    finally:
        # Cleanup
        detection_thread.stop()
        vs.stop()
        cv2.destroyAllWindows()
    
    return True


if __name__ == "__main__":
    data_pipeline_thread()