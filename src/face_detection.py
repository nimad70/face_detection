"""
Face Detection Utility Module

This module provides utility functions for webcam initialization, face detection using Haar Cascade classifiers,
frame preprocessing, bounding box scaling, and visualization of detected faces.
"""

import cv2
import sys


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


def preprocessing_frame(frame):
    """
    Preprocesses a video frame by resizing, converting to grayscale, enhancing contrast, and applying a Gaussian blur.
    
    Args:
        frame (numpy.ndarray): The original video frame.
    
    Returns:
        gray_frame (numpy.ndarray): The preprocessed frame.
    """
    resized_frame = cv2.resize(frame, (300, 300))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    gray_frame = cv2.equalizeHist(gray_frame) # To enhance contrast
    gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0) # To remove noises

    return gray_frame


def face_detector(classifier, frame):
    """
    Detects faces in a given frame using the Haar cascade classifier.
    
    Args:
        classifier (cv2.CascadeClassifier): The loaded Haar cascade classifier.
        frame (numpy.ndarray): The preprocessed frame.
    
    Returns:
        faces (list): A list of detected face bounding boxes (x, y, w, h).
    """
    faces = classifier.detectMultiScale(
        frame,
        scaleFactor=1.1, # scale down the image by 20% to detect larger faces
        minNeighbors=5, # number of neighboring rectangles (lower -> more false positives)
        minSize=(30, 30), # minimum size of the face to detect
        maxSize=(300, 300)  # Ignore very large faces
    )

    return faces


def scale_bounding_box(frame, detected_faces):
    """
    Scales the detected face bounding boxes back to the original frame size.
    
    Args:
        frame (numpy.ndarray): The original frame before resizing.
        detected_faces (list): List of detected face bounding boxes (x, y, w, h) in the resized frame.
    
    Returns:
        scaled_faces (list): List of rescaled bounding boxes (x, y, w, h) in the original frame size.
    """
    original_height, original_width = frame.shape[:2] # Get the original height and width of the frame
    scale_x = original_width / 300
    scale_y = original_height / 300

    scaled_faces = [] # Store the coordinates of the detected faces
    for (x, y, w, h) in detected_faces:
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        scaled_faces.append((x, y, w, h))
    
    return scaled_faces


def draw_rectangle(frame, detected_faces):
    """
    Draws bounding boxes (rectangles) around detected faces on the original frame.
    
    Args:
        frame (numpy.ndarray): The original video frame.
        detected_faces (list): List of detected face bounding boxes (x, y, w, h).
    """
    scaled_faces = scale_bounding_box(frame, detected_faces)

    for (x, y, w, h) in scaled_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def main():
    """
    Captures video from the webcam, detects faces, and displays the processed frames with bounding boxes from 
    face_detection.py module.
    
    The function continuously reads frames from the webcam, preprocesses them, detects faces using Haar Cascade classifier,
    and draws bounding boxes around detected faces. The video stream is displayed in a window, and the loop exits
    when the user presses 'q'.
    """
    cap = video_capture()

    while True:
        res, frame = cap.read() # capture frames from the video source

        # If the frame is not captured, then break the loop
        if not res:
            print("Failed to capture frames")
            break

        preprocessed_frame = preprocessing_frame(frame)
        face_classifier = haarcascade_classifier()
        detected_faces = face_detector(face_classifier, preprocessed_frame)
        draw_rectangle(frame, detected_faces)

        cv2.imshow("Webcam", frame) # Display the captured frame

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
