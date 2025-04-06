"""
This module provides a class to handle video streaming from a camera or video file using OpenCV.
It uses threading to continuously read frames from the video source in the background, allowing for real-time processing of video frames.
"""

from threading import Thread

import cv2


class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception("Could not open video device")
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame
    
    def stop(self):
        self.stopped = True
        self.stream.release()
