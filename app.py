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
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # To enhance contrast
    gray = cv2.equalizeHist(gray)

    # To remove noises
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray



if __name__ == "__main__":
    
    cap = video_capture()

    # Continuously read the frames from the webcam
    while True:
        # Read a frame
        res, frame = cap.read()

        # If the frame is not captured, then break the loop
        if not res:
            print("Fail to capture frames")
            break

        gray_frame = preprocessing_frame(frame)

        # Display the captured frame
        # cv2.imshow("Webcam", frame)
        cv2.imshow("Webcam", gray_frame)

        # Exit the program if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()