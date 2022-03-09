from threading import Thread
import cv2
import dlib
import math

#class will check a) if there are faces b) face coords of two frames c) returns a yes if they match
        
class CheckFaceLoc:

    def __init__(self, capture1=None, capture2=None):
        self.capture1 = capture1
        self.capture2 = capture2
        self.stopped = False
        self.face_locations = []
        self.match = False

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:
            # Grab frames from live video streams
            _frame1 = self.capture1.read()
            _frame2 = self.capture2.read()

            # Make thme black and white for analysis
            _gray = cv2.cvtColor(_frame1, cv2.COLOR_BGR2GRAY)
            _gray2 = cv2.cvtColor(_frame2, cv2.COLOR_BGR2GRAY)

            #Find faces
            detector = dlib.get_frontal_face_detector()

            _faces= detector(_gray)
            _faces2= detector(_gray2)

            #Find face coords

            for _face in _faces:
                _faceleft = _face.left()
                _facetop = _face.top()
                _faceright = _face.right()
                _facebottom = _face.bottom()
                cv2.rectangle(_frame1, (_faceleft, _facetop), (_faceright, _facebottom), (0, 255, 0), 3)

            for _face2 in _faces2:
                _faceleft2 = _face2.left()
                _facetop2 = _face2.top()
                _faceright2 = _face2.right()
                _facebottom2 = _face2.bottom()
                cv2.rectangle(_frame2, (_faceleft2, _facetop2), (_faceright2, _facebottom2), (0, 255, 0), 3)

            #Are they a match?

            _closenessleft = math.isclose(_faceleft, _faceleft2, abs_tol = 70) #5 pixels
            _closenesstop = math.isclose(_facetop, _facetop2, abs_tol = 70)
            _closenessright = math.isclose(_faceright, _faceright2, abs_tol = 70)
            _closenessbottom = math.isclose(_facebottom, _facebottom2, abs_tol = 70)

            if _closenessleft == True and _closenesstop == True and _closenessright == True and _closenessbottom == True:
                print("match") #replace this
                _match = True
            else:
                _match = False
            
            self.match = _match

    def stop(self):
         self.stopped = True


