from multiprocessing import process, Queue
from src.object_detection.resNet_face_detector import load_detector, detect_faces, draw_rectangle
from src.utils.config import resNetSSD_config


def detection_worker(input_q, output_q):
    """
    Load the ResNet detector
    """
    gpu_enabled, prototxt, caffemodel, confidence_threshold, img_size = resNetSSD_config()
    net = load_detector(prototxt, caffemodel, gpu_enabled)

    while True:
        # if not input_q.empty():
        frame = input_q.get()
        if frame is None:
            break

        faces = detect_faces(frame, net, confidence_threshold, img_size)
        output_q.put((frame, faces))
