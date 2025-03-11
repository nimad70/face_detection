"""
Main Application Script for Webcam-Based Face Detection and Dataset Creation

This script integrates functionalities from various modules to capture webcam video, detect faces,
create labeled datasets, and split data into training, validation, and testing sets.
"""

import cv2
import sys
from src.face_detection import video_capture, haarcascade_classifier, face_detector, draw_rectangle, preprocessing_frame
from src.face_detection_multi_threading import start_threads
from src.dataset_creation import split_dataset, LABELS
from src.data_augmentation import save_augmented_images


def main():
    """
    Captures video from the webcam, detects faces, and displays the processed frames with bounding boxes from 
    face_detection.py module.
    
    The function continuously reads frames from the webcam, preprocesses them, detects faces using Haar Cascade classifier,
    and draws bounding boxes around detected faces. The video stream is displayed in a window, and the loop exits
    when the user presses 'q'.
    """
    cap = video_capture()
    classifier = haarcascade_classifier()

    while True:
        res, frame = cap.read() # capture frames from the video source

        # If the frame is not captured, then break the loop
        if not res:
            print("Fail to capture frames")
            break

        processed_frame = preprocessing_frame(frame)
        classifier = haarcascade_classifier()
        detected_faces = face_detector(classifier, processed_frame)
        draw_rectangle(frame, detected_faces)

        cv2.imshow("Webcam", frame) # Display the captured frame

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    # start_threads()
    is_splited = True
    while is_splited:
        start_threads()

        # To split the dataset
        res = split_dataset(LABELS)

        if res:
            break
        
        warning_text = "*WARNING: Please take at least 6 shots per each category to add to the dataset"
        
        print('-' * len(warning_text))
        print(f"{warning_text}")
        print('-' * len(warning_text))
        
    print("Smile and Non-smile datasets are splited")

    save_augmented_images()
    print("Augmented images are saved to the corresponding training datasets")
