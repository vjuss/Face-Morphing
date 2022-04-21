from threading import Thread
import cv2
import time

#edited to follow https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.update, args=()).start()
        return self

    def update(self):

        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()
            


    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True