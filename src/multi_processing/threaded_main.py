import cv2
from src.multi_processing.video_stearm import VideoStream
from src.multi_processing.threading_detection_worker import DetectionThread
from src.object_detection.resNet_face_detector import load_detector, detect_faces, draw_rectangle
from src.utils.config import resNetSSD_config

# Shared variables
current_frame = None
detected_faces = []

def get_frame():
    """
    Get the current frame from the video stream.
    """
    return current_frame


def set_result(frame, faces):
    """
    Set the detection results.
    """
    global detected_faces
    detected_faces = faces


def thread_main():
    """
    Main function to run the threaded face detection using ResNet model.
    """
    global current_frame

    vs = VideoStream().start()
    
    # Load the detector
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size = resNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)

    # Wrap the detection function
    detector = lambda frame: detect_faces(frame, net)
    
    # Create and start the detection thread
    detection_thread = DetectionThread(detector, get_frame, set_result)
    detection_thread.start()

    try:
        while True:
            ret, frame = vs.read()
            if not ret:
                break

            current_frame = frame.copy()

            # Draw bounding boxes around detected faces
            draw_rectangle(frame, detected_faces)

            cv2.imshow("Threaded Face Detection", current_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Cleanup
        detection_thread.stop()
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    thread_main()