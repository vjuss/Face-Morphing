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
#add a model that recognises when face is diff from other face and explore when it only sees one face!!
#replace eyes with other persons eyes: try spliot scanning
#add glitches from third webcam (observer of both peple, space) to the mix
#blurrinf etc transition effects
# goal is to make eyes align
#sound
# potentially add GAN to generate nww faces. CAN ALSO REPLACE DELANAY OIN THIS CODE
# one helpful sourcve for threading has been https://github.com/jpark7ca/face_recognition/blob/master/face_recognition_webcam_mt.py
# next: make delanay into a thread and class, check the tutorial above for performance
# pupil code bit https://stackoverflow.com/questions/67362053/is-there-a-way-to-select-a-specific-point-at-the-face-after-detecting-facial-lan



import cv2
from cv2 import FLOODFILL_FIXED_RANGE
import numpy as np
import dlib
import math
from AddDelaunay import AddDelaunay
from VideoGet import VideoGet
from CheckFaceLoc import CheckFaceLoc
from CheckFaces import CheckFaces
import random
import threading
import time

def countdown():
    global my_timer
    my_timer = 0

    for x in range(10):
        my_timer = my_timer + 1
        print(my_timer)
        sleep(1)

    print("10 seconds up")

def extract_index_nparray(nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

def main():

    video_capture2 = VideoGet(src=0).start()  #0 and 1 at home, 0 and 2 at uni
    video_capture = VideoGet(src=2).start()

    facedetector = dlib.get_frontal_face_detector()
    landmarkpredictor = dlib.shape_predictor("data/68_face_landmarks.dat")
    
    video_process= CheckFaceLoc(capture1 = video_capture, capture2=video_capture2, detector=facedetector, predictor=landmarkpredictor).start()
    #FIX THIS SO THAT YOU ONLY USE CERATIN PROPERTIES FROM SAME VIDOE PROCESS CLASS. CREATE FUNCTIONS THERE

    #for handling old sequences
    pastframes1 = list()
    pastframes2 = list()

    matchresult = False # placeholder when starting

    start = time.time()


    while True:
        if video_capture.stopped:
                video_capture.stop()
                break

        vidframe = video_capture.frame
        vidframe2 = video_capture2.frame

        two_faces = video_process.twofaces #checks if true or false 

        if two_faces == True:
            #vidframe = video_capture2.frame
            #vidframe2 = video_capture.frame
            #FLIPPING AND EYE / MATCH DETETCTUON WILL HAPPEN IN THIS CASE
            matchresult = video_process.match # check if faces or eyes match - make thi into a function or class
        else:
            #idframe = video_capture.frame
            #vidframe2 = video_capture2.frame
            matchresult = False

        if matchresult == True:
            print("faces match")
            matchtime = time.time() - start 
            print(matchtime)  #this number grows every time loop is run
            currenttime = time.time() -matchtime - start
            print(currenttime)
           
            pastframes1.append(vidframe) # storing ghost images to be used later
            pastframes2.append(vidframe2) #
        
            #WE WANT TO TRIGGER TIMER 
            #IF TIMER LESS THEN 20S, USE DELAUNAY EFFECT 1
            #IF TIMER BETWEEN 20S AND 40S, USE DELAUNAY EFFECT 2
            #IF TIMER BETWEEN 40-60S, USE DELAUNAY EFFECT 3 

            #these if statements are to be replaced with  a timer and diff delaunay functions / classes

            if len(pastframes1) > 20 and len(pastframes1) < 30: #testing ghost images on one
                vidframe = (random.choice(pastframes1))

            if len(pastframes1) > 30 and len(pastframes1) < 40: #testing ghost images on both
                vidframe = (random.choice(pastframes1))
                vidframe2 = (random.choice(pastframes2))

            if len(pastframes1) > 40 and len(pastframes1) < 60: #delanay bridges same person in diff time. maybe go through all the stored frames backwards
                vidframe = (random.choice(pastframes1))
                vidframe2 = video_capture.frame


            #THESE 3 LINES ARE THE GOAL, NOT WORKING YET
             #delaunay_process = AddDelaunay(frame = vidframe, frame2 = vidframe2, detector=facedetector, predictor = landmarkpredictor).start()
             #resultframe = delaunay_process.seamlessclone
             #resultframe2 = delaunay_process.seamlessclone2

            #gray = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
            #gray2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

            gray = cv2.cvtColor(video_process.frame, cv2.COLOR_BGR2GRAY) 
            gray2 = cv2.cvtColor(video_process.frame2, cv2.COLOR_BGR2GRAY) 

            mask = np.zeros_like(gray)
            height, width, channels = video_process.frame2.shape #was vidframe2
            img2_new_face = np.zeros((height, width, channels), np.uint8)
            indexes_triangles = []

            mask2 = np.zeros_like(gray2)
            height2, width2, channels2 = video_process.frame.shape #change all vidframes to vid process frameiksi to test
            img1_new_face = np.zeros((height2, width2, channels2), np.uint8)
            indexes_triangles2 = []

            #landmarks of first face

            faces = video_process.faces 
            faces2 = video_process.faces2 

            for face in faces:
                landmarks = video_process.landmarks
                landmarks_points = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x, y))

                points = np.array(landmarks_points, np.int32)

                convexhull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, convexhull, 255)    
                
                face_image_1 = cv2.bitwise_and(video_process.frame, video_process.frame, mask=mask)
        

                # Delaunay triangulation

                rect = cv2.boundingRect(convexhull)
                subdiv = cv2.Subdiv2D(rect)
                subdiv.insert(landmarks_points)
                triangles = subdiv.getTriangleList()
                triangles = np.array(triangles, dtype=np.int32)
                
                #indexes_triangles = []

                for t in triangles:
                    pt1 = (t[0], t[1])
                    pt2 = (t[2], t[3])
                    pt3 = (t[4], t[5])

                    index_pt1 = np.where((points == pt1).all(axis=1))
                    index_pt1 = extract_index_nparray(index_pt1)

                    index_pt2 = np.where((points == pt2).all(axis=1))
                    index_pt2 = extract_index_nparray(index_pt2)

                    index_pt3 = np.where((points == pt3).all(axis=1))
                    index_pt3 = extract_index_nparray(index_pt3)
                    
                    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                        triangle = [index_pt1, index_pt2, index_pt3]
                        indexes_triangles.append(triangle)

            # Face 2
            for face in faces2:
                landmarks2 = video_process.landmarks2
                landmarks_points2 = []
                for n in range(0, 68):
                    x = landmarks2.part(n).x
                    y = landmarks2.part(n).y
                    landmarks_points2.append((x, y))


                points2 = np.array(landmarks_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)
                cv2.fillConvexPoly(mask2, convexhull2, 255)  
                face_image_2 = cv2.bitwise_and(video_process.frame2, video_process.frame2, mask=mask2)


            # Creating empty mask
            lines_space_mask = np.zeros_like(gray)
            lines_space_new_face = np.zeros_like(video_process.frame2)

            # Triangulation of both faces
            for triangle_index in indexes_triangles:
                # Triangulation of the first face
                tr1_pt1 = landmarks_points[triangle_index[0]]
                tr1_pt2 = landmarks_points[triangle_index[1]]
                tr1_pt3 = landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = video_process.frame[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)


                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

                # Lines space
                cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
                cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
                cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
                lines_space = cv2.bitwise_and(video_process.frame, video_process.frame, mask=lines_space_mask)

                # Triangulation of second face
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2

                cropped_tr2_mask = np.zeros((h, w), np.uint8)

                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                # Reconstructing destination face
                img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

            #Face swapped (putting 1st face into 2nd face)
            img2_face_mask = np.zeros_like(gray2)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 100) #255 is full opacity
            img2_head_mask2 = cv2.fillConvexPoly(img2_face_mask, convexhull2, 220) #255 is full opacity
            img2_face_mask = cv2.bitwise_not(img2_head_mask)
            img2_face_mask2 = cv2.bitwise_not(img2_head_mask2)

            img2_head_noface = cv2.bitwise_and(video_process.frame2, video_process.frame2, mask=img2_face_mask)
            img2_head_noface2 = cv2.bitwise_and(video_process.frame2, video_process.frame2, mask=img2_face_mask2)
            result = cv2.add(img2_head_noface, img2_new_face)
            result2 = cv2.add(img2_head_noface2, img2_new_face)

            # Creating seamless clone of two faces
            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
            seamlessclone = cv2.seamlessClone(result, video_process.frame2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
            seamlessclone = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)

            seamlessclone2 = cv2.seamlessClone(result2, video_process.frame2, img2_head_mask2, center_face2, cv2.NORMAL_CLONE)
            seamlessclone2 = cv2.cvtColor(seamlessclone2, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("result", seamlessclone) #

            resultframe = seamlessclone
            resultframe2 = seamlessclone2

        else:
            resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
            if len(video_process.rightpupils) ==2:
                cv2.circle(resultframe, video_process.rightpupils, 20, (255), -1)
                cv2.circle(resultframe, video_process.leftpupils, 20, (255), -1)
            resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 
            if len(video_process.rightpupils2) ==2:
                cv2.circle(resultframe2, video_process.rightpupils2, 20, (255), -1)
                cv2.circle(resultframe2, video_process.leftpupils2, 20, (255), -1)

        outputs = np.concatenate((resultframe, resultframe2), axis=0) 
        cv2.imshow("Result", outputs) #CURRENT ERROR COMES FROM HERE: error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'imshow'

        key = cv2.waitKey(1)
        if key == 27: #esc
            break

    # Release handle to the webcam
    video_capture.stop()
    video_capture2.stop()
    video_process.stop()
    #delaunay_process.stop()

    cv2.destroyAllWindows()

main()

