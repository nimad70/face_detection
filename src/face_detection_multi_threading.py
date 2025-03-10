import cv2
import sys
import queue
import threading


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


# Continuously capture frames from the webcam and add them to the frame queue.
def capture_frames(cap, frame_queue):
    while True:
        # Capture frames from the video source
        res, frame = cap.read()

        # If the frame is not captured, then break the loop
        if not res:
            break

        # Store the frame in the queue
        if not frame_queue.full():
            frame_queue.put(frame)


if __name__ == "__main__":
    cap = video_capture()
    face_cascade_classifier = haarcascade_classifier()

    
    # To display the final frames
    # while True:
        # res, frame = cap.read()

        # if not res:
        #     print("Failed to capture frames")
        #     break

        # Display the resulting frame
        # cv2.imshow('Webcam face detection', frame)

        # Break the loop when 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break   


    # Release the video capture object
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()