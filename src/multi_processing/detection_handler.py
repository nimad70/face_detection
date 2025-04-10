"""
This module handles the detection of faces in images using a ResNet SSD detector.
It utilizes multiprocessing to improve performance by processing frames in parallel.
"""

from multiprocessing import Process, Queue

from src.object_detection.resNet_face_detector import (
    load_detector, 
    detect_faces, 
    draw_rectangle,
)
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
