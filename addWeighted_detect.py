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



from cgitb import handler
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

    video_capture2 = VideoGet(src=1).start()  #0 and 1 at home, 0 and 2 at uni
    video_capture = VideoGet(src=0).start()

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
        #
        # 
        #
        #  
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
            #
            #
            #
            #
            #

            if len(pastframes1) <= 30: #EFFECT 1. later: timer less than 20 s
                #SOLVE NEXT: TWO DIFF OUTCOMES WITH THEIR OWN BACKGROUDS: NOW SAME FRAME IN BOTH RESULTS
                sourceframe = video_process.frame
                destinationframe = video_process.frame2

                print("effect 1")

                sourcegray = cv2.cvtColor(sourceframe, cv2.COLOR_BGR2GRAY) 
                destinationgray = cv2.cvtColor(destinationframe, cv2.COLOR_BGR2GRAY) 

                sourcemask = np.zeros_like(sourcegray)
                height, width, channels = destinationframe.shape #was vidframe2
                height2, width2, channels2 = sourceframe.shape
                destination_new_face = np.zeros((height, width, channels), np.uint8)
                source_new_face = np.zeros((height2, width2, channels2), np.uint8)
                indexes_triangles = []

                destinationmask = np.zeros_like(destinationgray)
                height2, width2, channels2 = sourceframe.shape 

                #landmarks of first face

                sourcefaces = video_process.faces 
                destinationfaces = video_process.faces2 

                for sourceface in sourcefaces:
                    sourcelandmarks = video_process.landmarks
                    sourcelandmarks_points = []
                    for n in range(0, 68):
                        x = sourcelandmarks.part(n).x
                        y = sourcelandmarks.part(n).y
                        sourcelandmarks_points.append((x, y))

                    sourcefacepoints = np.array(sourcelandmarks_points, np.int32)

                    sourceconvexhull = cv2.convexHull(sourcefacepoints)
                    cv2.fillConvexPoly(sourcemask, sourceconvexhull, 255)    
                    

                    # Delaunay triangulation

                    sourcerect = cv2.boundingRect(sourceconvexhull)
                    sourcesubdiv = cv2.Subdiv2D(sourcerect)
                    sourcesubdiv.insert(sourcelandmarks_points)
                    sourcetriangles = sourcesubdiv.getTriangleList()
                    sourcetriangles = np.array(sourcetriangles, dtype=np.int32)
                    
                    #indexes_triangles = []

                    for t in sourcetriangles:
                        pt1 = (t[0], t[1])
                        pt2 = (t[2], t[3])
                        pt3 = (t[4], t[5])

                        index_pt1 = np.where((sourcefacepoints == pt1).all(axis=1))
                        index_pt1 = extract_index_nparray(index_pt1)

                        index_pt2 = np.where((sourcefacepoints == pt2).all(axis=1))
                        index_pt2 = extract_index_nparray(index_pt2)

                        index_pt3 = np.where((sourcefacepoints == pt3).all(axis=1))
                        index_pt3 = extract_index_nparray(index_pt3)
                        
                        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                            sourcetriangle = [index_pt1, index_pt2, index_pt3]
                            indexes_triangles.append(sourcetriangle)

                # Face 2
                for destinationface in destinationfaces:
                    destinationlandmarks = video_process.landmarks2
                    destinationlandmarks_points = []
                    for n in range(0, 68):
                        x = destinationlandmarks.part(n).x
                        y = destinationlandmarks.part(n).y
                        destinationlandmarks_points.append((x, y))


                    destinationpoints = np.array(destinationlandmarks_points, np.int32)
                    destinationconvexhull = cv2.convexHull(destinationpoints)
                    cv2.fillConvexPoly(destinationmask, destinationconvexhull, 255)  


                # Creating empty mask
                source_lines_space_mask = np.zeros_like(sourcegray)
                destination_lines_space_mask = np.zeros_like(destinationgray)


                # Triangulation of both faces, NO NEED TO DO TWICE
                for triangle_index in indexes_triangles:
                    # Triangulation of the first face
                    tr1_pt1 = sourcelandmarks_points[triangle_index[0]]
                    tr1_pt2 = sourcelandmarks_points[triangle_index[1]]
                    tr1_pt3 = sourcelandmarks_points[triangle_index[2]]
                    sourcetriangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


                    rect1 = cv2.boundingRect(sourcetriangle)
                    (x, y, w, h) = rect1
                    (xu, yu, wu, hu) = rect1
                    cropped_triangle = sourceframe[y: yu + hu, x: xu + wu]
                    cropped_tr1_mask = np.zeros((h, w), np.uint8)


                    sourcefacepoints = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                    [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                    [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr1_mask, sourcefacepoints, 255)

                    # Lines space
                    cv2.line(source_lines_space_mask, tr1_pt1, tr1_pt2, 255)
                    cv2.line(source_lines_space_mask, tr1_pt2, tr1_pt3, 255)
                    cv2.line(source_lines_space_mask, tr1_pt1, tr1_pt3, 255)

                    # Triangulation of second face
                    tr2_pt1 = destinationlandmarks_points[triangle_index[0]]
                    tr2_pt2 = destinationlandmarks_points[triangle_index[1]]
                    tr2_pt3 = destinationlandmarks_points[triangle_index[2]]
                    destinationtriangle = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


                    rect2 = cv2.boundingRect(destinationtriangle)
                    (x, y, w, h) = rect2


                    cropped_tr2_mask = np.zeros((h, w), np.uint8)

        
                    cropped_triangle_source = sourceframe[y: y + h, x: x + w] #OR dest frame?
                    cropped_tr_mask_source = np.zeros((h, w), np.uint8)  #MASK NOT CORRECT YET
    

                    

                    destinationpoints = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr2_mask, destinationpoints, 255)

                    # Lines space vol 2
                    cv2.line(destination_lines_space_mask, tr2_pt1, tr2_pt2, 255)
                    cv2.line(destination_lines_space_mask, tr2_pt2, tr2_pt3, 255)
                    cv2.line(destination_lines_space_mask, tr2_pt1, tr2_pt3, 255)


                    # Warp triangles
                    sourcefacepoints = np.float32(sourcefacepoints)
                    destinationpoints = np.float32(destinationpoints)
                    M = cv2.getAffineTransform(sourcefacepoints, destinationpoints)
                    M2 = cv2.getAffineTransform(destinationpoints, sourcefacepoints)
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                    warped_triangle_source = cv2.warpAffine(cropped_triangle_source, M, (w, h)) #THESE NOT CORRECT YET: CROPPED, M, w, h
                    warped_triangle_source = cv2.bitwise_and(warped_triangle_source, warped_triangle_source, mask=cropped_tr_mask_source)

                
                    # Reconstructing destination face
                    destination_new_face_rect_area = destination_new_face[y: y + h, x: x + w]
                    destination_new_face_rect_area_gray = cv2.cvtColor(destination_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                    _, mask_triangles_designed = cv2.threshold(destination_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                    destination_new_face_rect_area = cv2.add(destination_new_face_rect_area, warped_triangle)
                    destination_new_face[y: y + h, x: x + w] = destination_new_face_rect_area

                    # Reconstructing source face

                    source_new_face_rect_area = source_new_face[y: y + h, x: x + w]    #H, W ETC need editing. #THESE NOT CORRECT YET
                    source_new_face_rect_area_gray = cv2.cvtColor(source_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                    _, mask_triangles_designed_source = cv2.threshold(source_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle_source = cv2.bitwise_and(warped_triangle_source, warped_triangle_source, mask=mask_triangles_designed_source)

                    source_new_face_rect_area = cv2.add(source_new_face_rect_area, warped_triangle_source)
                    source_new_face[y: y + h, x: x + w] = source_new_face_rect_area


                #Face swapped (putting 1st face into 2nd face)

                opacity = len(pastframes1) * 8 #max value is 24

                destination_swapped_face_mask = np.zeros_like(destinationgray)
                source_swapped_face_mask = np.zeros_like(sourcegray)
                #img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 155) 
                destination_head_mask = cv2.fillConvexPoly(destination_swapped_face_mask, destinationconvexhull, opacity) 
                source_head_mask_new = cv2.fillConvexPoly(source_swapped_face_mask, sourceconvexhull, opacity) #EDITED
                destination_swapped_face_mask = cv2.bitwise_not(destination_head_mask)
                source_swapped_face_mask = cv2.bitwise_not(source_head_mask_new)

                destination_head_noface = cv2.bitwise_and(destinationframe, destinationframe, mask=destination_swapped_face_mask)
                source_head_noface_new = cv2.bitwise_and(sourceframe, sourceframe, mask=source_swapped_face_mask)
                result = cv2.add(destination_head_noface, destination_new_face)
                result2 = cv2.add(source_head_noface_new , source_new_face) 
    

                # Creating seamless clone of two faces
                (x, y, w, h) = cv2.boundingRect(destinationconvexhull)
                center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                (x2, y2, w2, h2) = cv2.boundingRect(sourceconvexhull)
                center_face_source = (int((x2 + x2 + w2) / 2), int((y2 + y2 + h2) / 2))
                seamlessclone = cv2.seamlessClone(result, destinationframe, destination_head_mask, center_face2, cv2.NORMAL_CLONE)
                seamlessclone = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)

                seamlessclone2 = cv2.seamlessClone(result2, sourceframe, source_head_mask_new, center_face_source, cv2.NORMAL_CLONE)  #EDITED
                seamlessclone2 = cv2.cvtColor(seamlessclone2, cv2.COLOR_BGR2GRAY)

                resultframe = seamlessclone
                resultframe2 = seamlessclone2

            #
            #
            #
            #
            #
            #


            elif len(pastframes1) > 60 and len(pastframes1) <= 90: #EFFECT 2. later: timer less than 40s 
                print("effect 2")
                resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

                #for both participants, delaunay becomes time travel between their own frames. Testing with one first

                



                


                














    
            #
            #
            #
            #
            #
            #
            #
            elif len(pastframes1) > 90 and len(pastframes1) <= 120: #EFFECT 3. later: timer less than 60s 
                print("effect 3")
                resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

            else:
                print("timer full, reset the sketch")
                resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY)


            #THESE 3 LINES ARE THE GOAL, NOT WORKING YET
             #delaunay_process = AddDelaunay(frame = vidframe, frame2 = vidframe2, detector=facedetector, predictor = landmarkpredictor).start()
             #resultframe = delaunay_process.seamlessclone
             #resultframe2 = delaunay_process.seamlessclone2

            #gray = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
            #gray2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 



        #
        #
        #
        #
        #

        else: # if match result not true, just draw eyes
            resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
            resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

            if len(video_process.rightpupils) ==2 and len(video_process.rightpupils2) ==2:
                cv2.circle(resultframe2, video_process.rightpupils, 20, (0), -1) #drawing eyes to opponent's frame
                cv2.circle(resultframe2, video_process.leftpupils, 20, (0), -1)
                cv2.circle(resultframe, video_process.rightpupils2, 20, (255), -1)
                cv2.circle(resultframe, video_process.leftpupils2, 20, (255), -1)

            else:
                print("not drawing eyes")

        outputs = np.concatenate((resultframe, resultframe2), axis=0) 
        cv2.imshow("Result", outputs) 

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

