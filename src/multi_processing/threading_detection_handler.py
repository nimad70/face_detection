"""
This module contains a class that runs a face detection algorithm in a separate thread.
It retrieves frames from a video stream and processes them using the detector. The results are set using a provided function.
It is designed to be used in a multi-threaded environment for real-time face detection.
"""

import time
import threading


class DetectionThread(threading.Thread):
    """
    A thread that runs a face detection algorithm on frames from a video stream.
    It retrieves frames from a provided function, processes them using the detector,
    and sets the results using another provided function.
    """
    def __init__(self, detector, frame_getter, result_setter):
        super().__init__(daemon=True)
        self.detector = detector
        self.frame_getter = frame_getter # function returning current frame
        self.result_setter = result_setter # function accepting results
        self.running = True

    def run(self):
        while self.running:
            frame = self.frame_getter()
            if frame is not None:
                faces = self.detector(frame)
                self.result_setter(frame, faces)
            time.sleep(0.001) # prevent CPU spamming

    def stop(self):
        self.running = False
