import cv2
import sys

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

# Continuously read the frames from the webcam
while True:
    # Read a frame
    res, frame = cap.read()

    # If the frame is not read, exit the program
    if not res:
        print("Fail to capture frames")
        break

    # Display the captured frame
    cv2.imshow("Webcam", frame)

    # Exit the program if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()