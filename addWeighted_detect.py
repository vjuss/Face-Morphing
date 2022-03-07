#print "one face" if faces "match"
#play with addweighted value: this affewcst results, there could be two output results with different added weights and different detections 
#think about the effects: will the image be blurred, delanay, filter etc when match and what happens when that connection breaks? 
# will the resulting effect break the connection by blurring etc. making face not visibles
#one way to do it: introduce old frame sequences when faces match: what we see is no longer real time for a while and that has effect on detetcyopn
#study this for effects https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
#will the effect bridge time gaps instead of two people? or both


import cv2
import numpy as np
import dlib
from PIL import Image

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/68_face_landmarks.dat")

#eye coord variables for video 1

lefteyeX = 0 #38
lefteyeY = 0 #45
righteyeX = 0 
righteyeY = 0 

#eye coord variables for video 2

lefteyeX2 = 0
lefteyeY2 = 0
righteyeX2 = 0
righteyeY2 = 0


while True:

    ret1, frame = cap.read()
    ret2, frame2= cap2.read()

    if ret1==False or ret2==False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # blended image is here

    blended = cv2.addWeighted(frame,0.5,frame2,0.5,0)
    blendedgray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)


    faces= detector(gray)
    faces2= detector(gray2)
    faces3 = detector(blendedgray)
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray,face)

        for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                lefteyeX = landmarks.part(38).x #38
                lefteyeY = landmarks.part(38).y
                righteyeX = landmarks.part(45).x #45
                righteyeY = landmarks.part(45).y 
                print("feed1", lefteyeX, lefteyeY, righteyeX, righteyeY)


                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)


    for face2 in faces2:
        x1 = face2.left()
        y1 = face2.top()
        x2 = face2.right()
        y2 = face2.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks2 = predictor(gray2,face2)

        for n in range(0, 68):
                x = landmarks2.part(n).x
                y = landmarks2.part(n).y
                lefteyeX2 = landmarks2.part(38).x
                lefteyeY2 = landmarks2.part(38).y
                righteyeX2 = landmarks2.part(45).x
                righteyeY2 = landmarks2.part(45).y
                print("feed2", lefteyeX2, lefteyeY2, righteyeX2, righteyeY2)
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

    #if a match..do sth here





    # for face3 in faces3:
    #     x1 = face3.left()
    #     y1 = face3.top()
    #     x2 = face3.right()
    #     y2 = face3.bottom()
    #     #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #     landmarks3 = predictor(blendedgray,face3)

    #     for n in range(0, 68):
    #             x = landmarks3.part(n).x
    #             y = landmarks3.part(n).y
    #             cv2.circle(frame, (x, y), 6, (0, 0, 255), -1) #drawing ghost faces from the blended frame that is not visible

    cv2.imshow("Frame", frame) #show the other video feed only
    #cv2.imshow("Blended", blended) #show the blended result but draw faces from raw feeds as well 

    key = cv2.waitKey(1)
    if key == 27: #esc
        break


# cv2.destroyAllWindows() # destroys the window showing image