#steps for first MVP
#1)alert when faces match (rect fine first) DONE
#2)add two output feeds DONE
#3)test effetc: make addweighted a variable that changes on outputs when faces match in the og DONE
#4)introduce old frame sequences to outputs when faces match: 
# what we see is no longer real time for a while and that has effect on detetcyopn
#5)add timing to the effect if needed: will it continue until faces matc gain / 10 secs etv
#6)delanay etc blurring can make it difficult to detect whose face it is and thus match. it can gap two people or one person in diff times
#7)improve performance: try threads and maybe switch from python to openframeworks or use lighter face detection model

#IDEAS
#maybe face analysis is being done on outputs at some point instead
#play with addweighted value(s;ider): this can make your own face disappear and the other appear, this affewcst results, there could be two output results with different added weights and different detections 
#think about the effects: will the image be blurred, delanay, filter etc when match and what happens when that connection breaks? 
# potentially touch the raw feed as well: will the resulting effect break the connection by blurring etc. making face not visibles
# in output feeds, delanay can make it difficult to detect whose face it is and thus match
#study this for effects https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
#will the effect bridge time gaps instead of two people? or both
#add a model that recognises when face is diff from other face and explore when it only sees one face
#replace eyes with other persons eyes: try spliot scanning
#add glitches from third webcam (observer of both peple, space) to the mix
#blurrinf etc transition effects
# goal is to make eyes align
#sound
# potentially add ml to generate nww faces


import cv2
import numpy as np
import dlib
from PIL import Image
import math


#for handling old sequences
pastframes1 = list()
pastframes2 = list()

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

is_recording = False
has_recorded = False

while True:

    ret1, frame = cap.read()
    ret2, frame2= cap2.read()

    if ret1==False or ret2==False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    faces= detector(gray)
    faces2= detector(gray2)
    
    for face in faces:
        faceleft = face.left()
        facetop = face.top()
        faceright = face.right()
        facebottom = face.bottom()
        cv2.rectangle(frame, (faceleft, facetop), (faceright, facebottom), (0, 255, 0), 3)
        #print("face 1", faceleft, facetop, faceright, facebottom)


    for face2 in faces2:
        faceleft2 = face2.left()
        facetop2 = face2.top()
        faceright2 = face2.right()
        facebottom2 = face2.bottom()
        cv2.rectangle(frame, (faceleft2, facetop2), (faceright2, facebottom2), (0, 255, 0), 3)
        #print("face 2", faceleft2, facetop2, faceright2, facebottom2)

    closenessleft = math.isclose(faceleft, faceleft2, abs_tol = 70) #5 pixels
    closenesstop = math.isclose(facetop, facetop2, abs_tol = 70)
    closenessright = math.isclose(faceright, faceright2, abs_tol = 70)
    closenessbottom = math.isclose(facebottom, facebottom2, abs_tol = 70)

    if closenessleft == True and closenesstop == True and closenessright == True and closenessbottom == True:
        print("faces match")
        is_recording = True
        has_recorded = True
        pastframes1.append(frame) #sequence of raw images, can be 100 for example. frame is the realtime image
        pastframes2.append(frame2) #

        weightframe1 = 0.7
        weightframe2 = 0.3

    else:
        is_recording = False
        weightframe1 = 0.3
        weightframe2 = 0.7

    #here an if condition to decide whether we use live feed or the saved loop

    if is_recording == False and has_recorded == True: #if something has been saved
        print("using old loop")
        
        displayframe2 = frame2 # use og
        print(len(pastframes1))

        for i in range (len(pastframes1)):
            displayframe1 = pastframes1[i]
            output1 = cv2.addWeighted(displayframe1, weightframe1,displayframe2, weightframe2, 0)
            output2 = cv2.addWeighted(displayframe1, weightframe2,displayframe2, weightframe1, 0)

    else:
        displayframe1 = frame # use og
        displayframe2 = frame2 # use og
        output1 = cv2.addWeighted(displayframe1, weightframe1,displayframe2, weightframe2, 0)
        output2 = cv2.addWeighted(displayframe1, weightframe2,displayframe2, weightframe1, 0)

    #two outputs for showing effects and feedback    

    outputs = np.concatenate((output1, output2), axis=0)

    cv2.imshow("Outputs", outputs) #showing input and detection

    #cv2.imshow("Blended", blended) #show the blended result but draw faces from raw feeds as well 

    key = cv2.waitKey(1)
    if key == 27: #esc
        break

cv2.destroyAllWindows()

