#steps for first MVP
#1)alert when faces match (rect fine first)
#2)add two output feeds
#3)test effetc: make addweighted a variable that changes on outputs when faces match in the og
#4)introduce old frame sequences to outputs when faces match: what we see is no longer real time for a while and that has effect on detetcyopn
#5)add timing to the effect if needed: will it continue until faces matc gain / 10 secs etv
#6)delanay etc blurring can make it difficult to detect whose face it is and thus match. it can gap two people or one person in diff times


#IDEAS
#maybe face analysis is being done on outputs at some point instead
#play with addweighted value: this affewcst results, there could be two output results with different added weights and different detections 
#think about the effects: will the image be blurred, delanay, filter etc when match and what happens when that connection breaks? 
# potentially touch the raw feed as well: will the resulting effect break the connection by blurring etc. making face not visibles
# in output feeds, delanay can make it difficult to detect whose face it is and thus match
#one way to do it: introduce old frame sequences when faces match: what we see is no longer real time for a while and that has effect on detetcyopn
#study this for effects https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
#will the effect bridge time gaps instead of two people? or both


import cv2
import numpy as np
import dlib
from PIL import Image
import math

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/68_face_landmarks.dat")


#face coord variables for video 1

faceleft = 0
facetop = 0
faceright = 0
facebottom = 0

#face coord variables for video 2

faceleft2 = 0
facetop2 = 0
faceright2 = 0
facebottom2 = 0


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
        faceleft = face.left()
        facetop = face.top()
        faceright = face.right()
        facebottom = face.bottom()
        cv2.rectangle(frame, (faceleft, facetop), (faceright, facebottom), (0, 255, 0), 3)
        print("face 1", faceleft, facetop, faceright, facebottom)
        # landmarks = predictor(gray,face)

        # for n in range(0, 68):
        #         x = landmarks.part(n).x
        #         y = landmarks.part(n).y
        #         lefteyeX = landmarks.part(38).x #38
        #         lefteyeY = landmarks.part(38).y
        #         righteyeX = landmarks.part(45).x #45
        #         righteyeY = landmarks.part(45).y 
        #         cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        #print("feed1", lefteyeX, lefteyeY, righteyeX, righteyeY)


    for face2 in faces2:
        faceleft2 = face2.left()
        facetop2 = face2.top()
        faceright2 = face2.right()
        facebottom2 = face2.bottom()
        cv2.rectangle(frame, (faceleft2, facetop2), (faceright2, facebottom2), (0, 255, 0), 3)
        print("face 2", faceleft2, facetop2, faceright2, facebottom2)
        # landmarks2 = predictor(gray2,face2)

        # for n in range(0, 68):
        #         x = landmarks2.part(n).x
        #         y = landmarks2.part(n).y
        #         lefteyeX2 = landmarks2.part(38).x
        #         lefteyeY2 = landmarks2.part(38).y
        #         righteyeX2 = landmarks2.part(45).x
        #         righteyeY2 = landmarks2.part(45).y
        #         cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

        #print("feed2", lefteyeX2, lefteyeY2, righteyeX2, righteyeY2)ß

    closenessleft = math.isclose(faceleft, faceleft2, abs_tol = 70) #5 pixels
    closenesstop = math.isclose(facetop, facetop2, abs_tol = 70)
    closenessright = math.isclose(faceright, faceright2, abs_tol = 70)
    closenessbottom = math.isclose(facebottom, facebottom2, abs_tol = 70)

    if closenessleft == True and closenesstop == True and closenessright == True and closenessbottom == True:
        print("face matches")

    # if lefteyeX ==lefteyeX2 and lefteyeY==lefteyeY2 and righteyeX==righteyeX2 and righteyeY==righteyeY2:
    #     print("eyes match exactly")


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
