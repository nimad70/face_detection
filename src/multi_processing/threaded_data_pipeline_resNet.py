"""
Threaded face detection using ResNet model.
This script captures video frames from a webcam, detects faces using a ResNet model, and displays the results in real-time.
It uses threading to handle the detection process separately from the video capture, allowing for smoother performance.
"""
import time
import csv
from pathlib import Path
import cv2
from src.multi_processing.video_stearm import VideoStream
from src.multi_processing.threading_detection_handler import DetectionThread
from src.object_detection.resNet_face_detector import load_detector, detect_faces, draw_rectangle
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
    print("\nThese are options you can choose from: \n\n"
    "1. Press 'q' to quit\n"
    "2. Press 's' to save the detected faces\n"
    "3. Press 'a' to save the detected faces without smile\n")


def data_pipeline_thread():
    """
    Main function to run the threaded face detection using ResNet model.
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
    
    display_menu()

    try:
        while True:
            ret, frame = vs.read()
            if not ret:
                break

            current_frame = frame.copy()

            # Draw bounding boxes around detected faces
            draw_rectangle(frame, detected_faces)

            cv2.imshow("Threaded Face Detection", current_frame)

            with open(DATASET_PATH / "labels.csv", mode="a", newline="") as csvfile:
                label_writer = csv.writer(csvfile) # Create a CSV writer object
                selected_key = cv2.waitKey(1) & 0xFF # Wait for the user to press a key
                
                # To save the detected faces inside smile/nosmile directories
                if selected_key == ord('s') or selected_key == ord('a'):
                    for (startX, startY, endX, endY) in detected_faces:
                        face = frame[startY+1:endY, startX+1:endX] # Crop the detected face
                        filename = f"face_{int(time.time())}.jpg" # Create a unique filename
                        label = "smile" if selected_key == ord('s') else 'nosmile' # Assign the label
                        # Save the face in the corresponding directory
                        path = SMILE_PATH / filename if selected_key == ord('s') else NO_SMILE_PATH /filename
                        cv2.imwrite(str(path), face) # Save the face image

                        # write the label in a CSV file
                        label_writer.writerow([filename, label])

                # Exit if the user presses 'q'
                if selected_key == ord('q'):
                    break
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

    finally:
        # Cleanup
        detection_thread.stop()
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    data_pipeline_thread()