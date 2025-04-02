import cv2
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception("Could not open video device")
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame
    
    def stop(self):
        self.stopped = True
        self.stream.release()
