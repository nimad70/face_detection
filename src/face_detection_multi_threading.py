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


# Process frames by converting them to grayscale, resizing, and applying preprocessing.
def process_frames(frame_queue, gray_queue):
    # Continuously process frames from the frame queue
    while True:
        if not frame_queue.empty():
            # retrieve the frame from the queue
            frame = frame_queue.get()

            # Store original size for scaling bounding boxes
            original_height, original_width = frame.shape[:2]

            # Resize the frame to 300x300
            resized_frame = cv2.resize(frame, (300, 300))

            # Convert to grayscale
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Store the processed frame in the queue
            if not gray_queue.full():
                gray_queue.put((frame, gray, original_width, original_height))



# Detect faces in processed frames and scale bounding boxes back to the original size.
def face_detection(gray_queue, faces_queue, face_cascade_classifier):
    while True:
        # If the queue is not empty, retrieve the frame and processed frames
        if not gray_queue.empty():
            frame, gray, original_width, original_height = gray_queue.get()

            # Detect faces in the resized (300x300) grayscale image
            faces = face_cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=7,
                minSize=(30, 30),
            )

            # Scale bounding boxes back to the original size
            scale_x = original_width / 300
            scale_y = original_height / 300

            # Store the scaled coordinates of the detected faces
            scaled_faces = []
            for (x, y, w, h) in faces:
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                scaled_faces.append((x, y, w, h))

            # Store the frame and scaled detected faces in the queue
            if not faces_queue.full():
                faces_queue.put((frame, scaled_faces))


# Start the video capture, frame processing, and face detection threads.
def start_threads():
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


    # To display the final frames
    while True:
        # If the queue is not empty, retrieve the frame and detected faces
        if not faces_queue.empty():
            frame, faces = faces_queue.get()

            # Draw a bounding box around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Webcam face detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   


    # Release the video capture object
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_threads()
    # cap = video_capture()
    # face_cascade_classifier = haarcascade_classifier()

    
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
    # cap.release()
    # # Close all windows
    # cv2.destroyAllWindows()