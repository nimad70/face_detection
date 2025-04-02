import cv2
from src.multi_processing.video_stearm import VideoStream
from src.object_detection.resNet_face_detector import load_detector, detect_faces, draw_rectangle
from src.utils.config import resNetSSD_config


def multi_processing_main():
    """
    Main function to run the multi-processing face detection using ResNet model.
    """
    # Initialize video stream and load the model
    vs = VideoStream().start()

    # Load the ResNet model for face detection
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size = resNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)

    while True:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces in the frame using the ResNet model
        faces = detect_faces(frame, net, confidence_threshold, img_size)
        draw_rectangle(frame, faces)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


