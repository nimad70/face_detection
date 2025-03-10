import cv2
import sys


# Create a VideeCapture function to capture the video from the webcam
def video_capture():
    # Open the first available webcam
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


def preprocessing_frame(frame):
    # Resize the frame to 300x300 for faster processing
    resized_frame = cv2.resize(frame, (300, 300))

    # Convert the image to grayscale for computational efficiency
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # To enhance contrast
    gray_frame = cv2.equalizeHist(gray_frame)

    # To remove noises
    gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    return gray_frame



if __name__ == "__main__":

    # load haarcascade classifier for face detection
    face_cascade_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Check if the classifier is loaded
    if face_cascade_classifier.empty():
        print("Face cascade classifier is empty")
        sys.exit()
    
    # Create a video capture object
    cap = video_capture()

    # Continuously read the frames from the webcam
    while True:
        # Read a frame
        res, frame = cap.read()

        # Get the original height and width of the frame to rescale the bounding box
        original_height, original_width = frame.shape[:2]

        # If the frame is not captured, then break the loop
        if not res:
            print("Fail to capture frames")
            break

        # Preprocess the frame
        processed_frame = preprocessing_frame(frame)

        # Detect faces in the image
        faces = face_cascade_classifier.detectMultiScale(
            processed_frame,
            scaleFactor=1.1, # scale down the image by 20% to detect larger faces
            minNeighbors=5, # number of neighboring rectangles (lower -> more false positives)
            minSize=(30, 30), # minimum size of the face to detect
            maxSize=(300, 300)  # Ignore very large faces
        )

        # Scale bounding boxes back to original size
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

        # Draw a bounding box around the detected faces
        for (x, y, w, h) in scaled_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the captured frame
        cv2.imshow("Webcam", frame)
        # cv2.imshow("Webcam", gray_frame)

        # Exit the program if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()