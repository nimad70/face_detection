import time
import cv2
from multiprocessing import Process, Queue
from src.multi_processing.video_stearm import VideoStream
from src.multi_processing.detection_worker import detection_worker
from src.object_detection.resNet_face_detector import draw_rectangle



def multi_processing_main():
    """
    Main function to run the multi-processing face detection using ResNet model.
    """
    input_q = Queue(maxsize=5)
    output_q = Queue(maxsize=5)

    # Start detction process
    detection_process = Process(target=detection_worker, args=(input_q, output_q))
    detection_process.start()
    
    # Initialize video stream and load the model
    vs = VideoStream().start()
    time.sleep(1.0)  # give camera warm-up time

    exit_requested = False

    while not exit_requested:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Send frame to detection process
        if not input_q.full():
            input_q.put(frame)

        # Get processed frame
        if not output_q.empty():
            processed_frame, faces = output_q.get()
            draw_rectangle(processed_frame, faces)
            cv2.imshow("Multiprocessed Face Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit_requested =True
    
    input_q.put(None)
    detection_process.join(timeout=1)
    
    vs.stop()
    cv2.destroyAllWindows()


