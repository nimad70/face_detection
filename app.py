import cv2
import sys
from src.face_detection import video_capture, haarcascade_classifier, face_detector, draw_rectangle, preprocessing_frame


# To start the face detection process
def main():
    # Create a video capture object
    cap = video_capture()
    classifier = haarcascade_classifier()

    # Continuously read the frames from the webcam
    while True:
        # Read a frame
        res, frame = cap.read()

        # If the frame is not captured, then break the loop
        if not res:
            print("Fail to capture frames")
            break

        # Preprocess the frame
        processed_frame = preprocessing_frame(frame)
        
        # Haarcascade clasifier object
        classifier = haarcascade_classifier()

        # Detect faces in the frame
        detected_faces = face_detector(classifier, processed_frame)

        # Draw a bounding box around the detected faces
        draw_rectangle(frame, detected_faces)

        # Display the captured frame
        cv2.imshow("Webcam", frame)
        # cv2.imshow("Webcam", gray_frame)

        # Exit the program if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    