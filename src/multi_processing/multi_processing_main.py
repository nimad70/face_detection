import cv2
from src.multi_processing.video_stearm import VideoStream
from src.object_detection.resNet_face_detector import run_face_detection

def main():
    vs = VideoStream().start()

    while True:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Face detection
        run_face_detection(frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


