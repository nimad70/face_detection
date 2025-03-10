import cv2
import sys
from src.face_detection import video_capture, haarcascade_classifier, face_detector, draw_rectangle, preprocessing_frame


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
    main()
    