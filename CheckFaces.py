from threading import Thread
import cv2
import dlib
import math
import numpy as np

#class will check if there are two faces
        
class CheckFaces:

    def __init__(self, capture1=None, capture2=None, detector=None):
        self.capture1 = capture1
        self.capture2 = capture2
        self.detector = detector
        self.stopped = False
        self.twofaces = False
        self.faces = ()
        self.faces2 = ()
        self.frame = None
        self.frame2 = None

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:
            # Grab frames from live video streams
            _frame1 = self.capture1.read()
            _frame2 = self.capture2.read()
            self.frame = _frame1
            self.frame2 = _frame2

            # Make thme black and white for analysis
            _gray = cv2.cvtColor(_frame1, cv2.COLOR_BGR2GRAY)
            _gray2 = cv2.cvtColor(_frame2, cv2.COLOR_BGR2GRAY)

            #Find faces
            _detector = self.detector

            _faces= _detector(_gray)
            _faces2= _detector(_gray2)

            #add: if both retun 1 on avg on 5 frame, retunr true

            if len(_faces)==1 and len(_faces2)==1:  #if both webcams detect one face, return true
                self.twofaces = True

            else:
                self.twofaces = False


    def stop(self):
         self.stopped = True

