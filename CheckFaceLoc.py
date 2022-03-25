from threading import Thread
import cv2
import dlib
import math
import numpy as np

#class will check a) if there are faces b) face coords of two frames c) returns a yes if they match
        
class CheckFaceLoc:

    def __init__(self, capture1=None, capture2=None, detector=None, predictor=None):
        self.capture1 = capture1
        self.capture2 = capture2
        self.detector = detector
        self.predictor = predictor
        self.stopped = False
        self.match = False
        self.faces = ()
        self.faces2 = ()
        self.twofaces = False
        self.frame = None
        self.frame2 = None
        self.leftpupils = ()
        self.rightpupils = ()
        self.leftpupils2 = ()
        self.rightpupils2 = ()
        self.landmarks = None

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
            _predictor = self.predictor

            _faces= _detector(_gray)
            _faces2= _detector(_gray2)
            self.faces = _faces
            self.faces2 = _faces2
            
            if len(_faces) == 1 and len(_faces2) == 1:
                self.twofaces = True
            else:
                self.twofaces = False

            #Find face coords

            for _face in _faces:
                _landmarks = _predictor(_gray, _face)
                self.landmarks = _landmarks
                _pupil_x = int((abs(_landmarks.part(39).x + _landmarks.part(36).x)) / 2) # The midpoint of a line Segment between eye's corners in x axis
                _pupil_y = int((abs(_landmarks.part(39).y + _landmarks.part(36).y)) / 2) # The midpoint of a line Segment between eye's corners in y axis
                _pupil_coordination = (_pupil_x, _pupil_y)
                self.leftpupils = _pupil_coordination

                _rpupil_x = int((abs(_landmarks.part(46).x + _landmarks.part(44).x)) / 2) # The midpoint of a line Segment between eye's corners in x axis
                _rpupil_y = int((abs(_landmarks.part(46).y + _landmarks.part(44).y)) / 2) # The midpoint of a line Segment between eye's corners in y axis
                _rpupil_coordination = (_rpupil_x, _rpupil_y)
                self.rightpupils = _rpupil_coordination


            for _face2 in _faces2:
                _landmarks2 = _predictor(_gray2, _face2)
                self.landmarks2 = _landmarks2
                _pupil_x2 = int((abs(_landmarks2.part(39).x + _landmarks2.part(36).x)) / 2) # The midpoint of a line Segment between eye's corners in x axis
                _pupil_y2 = int((abs(_landmarks2.part(39).y + _landmarks2.part(36).y)) / 2) # The midpoint of a line Segment between eye's corners in y axis
                _pupil_coordination2 = (_pupil_x2, _pupil_y2)
                self.leftpupils2 = _pupil_coordination2

                _rpupil_x2 = int((abs(_landmarks2.part(46).x + _landmarks2.part(44).x)) / 2) # The midpoint of a line Segment between eye's corners in x axis
                _rpupil_y2 = int((abs(_landmarks2.part(46).y + _landmarks2.part(44).y)) / 2) # The midpoint of a line Segment between eye's corners in y axis
                _rpupil_coordination2 = (_rpupil_x2, _rpupil_y2)
                self.rightpupils2 = _rpupil_coordination2

            #Are they a match?

            _closenesspupilsA = math.isclose(_rpupil_x, _rpupil_x2, abs_tol =70)
            _closenesspupilsB = math.isclose(_rpupil_y, _rpupil_y2, abs_tol =70)

            if _closenesspupilsA == True and _closenesspupilsB == True:
                _match = True

            else:
                _match = False
            
            self.match = _match


    def stop(self):
         self.stopped = True
