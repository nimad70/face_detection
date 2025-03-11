import sys
import cv2
import time
import csv
from pathlib import Path
import shutil
import random

import queue
import threading


# To automate the process of creating the dataset for smile and no smile images
# Path Constants
DATASET_PATH = Path("dataset")
SMILE_PATH = DATASET_PATH / "smile"
NO_SMILE_PATH = DATASET_PATH / "nosmile"

# To create directory if they don't exist already, create missing parents if needed
SMILE_PATH.mkdir(parents=True, exist_ok=True)
NO_SMILE_PATH.mkdir(parents=True, exist_ok=True)


# AUtomate spliting the data
# To create train/val/test datasets
DATA_PATH = Path("data")
TRAIN_DATA_PATH = DATA_PATH / "train"
VAL_DATA_PATH = DATA_PATH / "val"
TEST_DATA_PATH = DATA_PATH / "test"
LABELS = ["smile", "nosmile"]


for dirs in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]: 
    for label in LABELS:
        (dirs / label).mkdir(parents=True, exist_ok=True)

# https://docs.python.org/3/library/shutil.html
# To split the dataset
def split_dataset(label):
    # To define the ratio of the dataset
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    # To get the image files with from the asked directory
    image_files = list((DATASET_PATH / label).glob("*.jpg"))
    random.shuffle(image_files) # shuffle the images randomly
    print(f"Total {label} images: {len(image_files)}")

    train_count = int(len(image_files) * train_ratio)
    val_count = int(len(image_files) * val_ratio)

    for i, image in enumerate(image_files):
        if i < train_count:
            destination_path = TRAIN_DATA_PATH / label / image.name
        elif i < train_count + val_count:
            destination_path = VAL_DATA_PATH / label / image.name
        else:
            destination_path = TEST_DATA_PATH / label / image.name

        shutil.copy(str(image), str(destination_path))



# Milti-Threading/Processing
frame_queue = queue.Queue(maxsize=1) # Queue for raw frames
gray_queue = queue.Queue(maxsize=1) # Queue for preprocessed grayscale frames
faces_queue = queue.Queue(maxsize=1) # Queue for deteceted faces


def video_capture():
    """
    Opens the first available webcam and returns the VideoCapture object.
    
    Returns:
        cap (cv2.VideoCapture): The video capture object if a webcam is found.
    """
    cap = None
    for i in range(5):
        cap = cv2.VideoCapture(i)
        # Check if the webcam is opened
        if cap.isOpened():
            print(f"Using webcam number: {i}")
            break
        
    # In case no webcam is found to exit the program
    if not cap or not cap.isOpened():
        print("No webcam found!")
        sys.exit()
    
    return cap


def haarcascade_classifier():
    """
    Load the Haar Cascade classifier for face detection.
    
    Returns:
        face_cascade_classifier (cv2.CascadeClassifier): Preloaded face detection classifier.
    """
    face_cascade_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Check if the classifier is loaded
    if face_cascade_classifier.empty():
        print("Face cascade classifier is empty")
        sys.exit()

    return face_cascade_classifier


def capture_frames(cap, frame_queue):
    """
    Continuously capture frames from the webcam and add them to the frame queue.
    
    Args:
        cap (cv2.VideoCapture): Video capture object.
        frame_queue (queue.Queue): Queue to store captured frames.
    """
    while True:
        res, frame = cap.read() # Capture frames from the video source

        # If the frame is not captured, then break the loop
        if not res:
            break

        if not frame_queue.full():
            frame_queue.put(frame)


def process_frames(frame_queue, gray_queue):
    """
    Process frames by converting them to grayscale, resizing, and applying preprocessing.
    
    Args:
        frame_queue (queue.Queue): Queue containing raw frames.
        gray_queue (queue.Queue): Queue to store processed processed frames.
    """
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Store original size for scaling bounding boxes
            original_height, original_width = frame.shape[:2]
            
            resized_frame = cv2.resize(frame, (300, 300))
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
            gray = cv2.equalizeHist(gray) # To enhance contrast
            gray = cv2.GaussianBlur(gray, (3, 3), 0) # To remove noises

            if not gray_queue.full():
                gray_queue.put((frame, gray, original_width, original_height))


def face_detection(gray_queue, faces_queue, face_cascade_classifier):
    """
    Detect faces in processed frames and scale bounding boxes back to the original size.
    
    Args:
        gray_queue (queue.Queue): Queue containing processed frames.
        faces_queue (queue.Queue): Queue to store detected face.
        face_cascade_classifier (cv2.CascadeClassifier): Haar cascade classifier for face detection.
    """
    while True:
        # retrieve the frame and processed frames
        if not gray_queue.empty():
            frame, gray, original_width, original_height = gray_queue.get()
            
            # Detect faces in the processed frame
            faces = face_cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=7,
                minSize=(30, 30),
            )

            # Scale bounding boxes back to the original size
            scale_x = original_width / 300
            scale_y = original_height / 300

            scaled_faces = [] # Store the scaled coordinates of the detected faces
            for (x, y, w, h) in faces:
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                scaled_faces.append((x, y, w, h))

            if not faces_queue.full():
                faces_queue.put((frame, scaled_faces))


def start_threads():
    """
    Start the video capture, frame processing, and face detection threads.
    """
    cap = video_capture()
    face_cascade_classifier = haarcascade_classifier()

    # Video capture thread
    video_capture_thread = threading.Thread(
        target=capture_frames, 
        args=(cap, frame_queue), 
        daemon=True
    )

    # Frame processing thread
    frame_process_thread = threading.Thread(
        target=process_frames, 
        args=(frame_queue, gray_queue), 
        daemon=True)

    # Face detection thread
    face_detection_thread = threading.Thread(
        target=face_detection,
        args=((gray_queue, faces_queue, face_cascade_classifier)),
        daemon=True
    )

    # Start threads
    video_capture_thread.start()
    frame_process_thread.start()
    face_detection_thread.start()

    print("\nThese are options you can choose from: \n\n"
    "1. Press 'q' to quit\n"
    "2. Press 's' to save the detected faces\n"
    "3. Press 'a' to save the detected faces without smile")

    # To display the final frames
    while True:
        # Retrieve the frame and detected faces
        if not faces_queue.empty():
            frame, faces = faces_queue.get()

            # Draw a bounding box around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Webcam face detection', frame)

        # Exit if the user presses 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # To save detected faces
        with open(DATASET_PATH / "labels.csv", mode="a", newline="") as csvfile:
            label_writer = csv.writer(csvfile)
            selected_key = cv2.waitKey(1) & 0xFF

            if selected_key == ord('s') or selected_key == ord('a'):
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    filename = f"face_{int(time.time())}.jpg"
                    label = "smile" if selected_key == ord('s') else 'nosmile'
                    path = SMILE_PATH / filename if selected_key == ord('s') else NO_SMILE_PATH /filename
                    cv2.imwrite(str(path), face)

                    # To write the label in a CSV file
                    label_writer.writerow([filename, label])

            if selected_key == ord('q'):
                break

    # Release the video capture object
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()





if __name__ == "__main__":
    start_threads()
    # To split the dataset
    
    split_dataset("smile")
    print("Smile dataset split completed")