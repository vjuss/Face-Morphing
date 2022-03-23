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
from ctypes import wstring_at
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


            if len(pastframes1) <= 30: #EFFECT 1. later: timer less than 20 s

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


            if len(pastframes1) <= 30: #EFFECT 1. later: timer less than 20 s
                #SOLVE NEXT: TWO DIFF OUTCOMES WITH THEIR OWN BACKGROUDS: NOW SAME FRAME IN BOTH RESULTS
                print("effect 1")

                sourceframe = video_process.frame
                destinationframe = video_process.frame2

                height, width, channels = destinationframe.shape #was vidframe2
                height2, width2, channels2 = sourceframe.shape

                sourcegray = cv2.cvtColor(sourceframe, cv2.COLOR_BGR2GRAY) 
                destinationgray = cv2.cvtColor(destinationframe, cv2.COLOR_BGR2GRAY) 

                sourcemask = np.zeros_like(sourcegray)
                destinationmask = np.zeros_like(destinationgray)

                source_image_canvas = np.zeros((height2, width2, channels2), np.uint8)
                destination_image_canvas = np.zeros((height, width, channels), np.uint8)

                indexes_triangles = []

    
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

                    source_triangle_points = np.array(sourcelandmarks_points, np.int32)

                    sourceconvexhull = cv2.convexHull(source_triangle_points)
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

                        index_pt1 = np.where((source_triangle_points == pt1).all(axis=1))
                        index_pt1 = extract_index_nparray(index_pt1)

                        index_pt2 = np.where((source_triangle_points == pt2).all(axis=1))
                        index_pt2 = extract_index_nparray(index_pt2)

                        index_pt3 = np.where((source_triangle_points == pt3).all(axis=1))
                        index_pt3 = extract_index_nparray(index_pt3)
                        
                        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                            source_triangle = [index_pt1, index_pt2, index_pt3]
                            indexes_triangles.append(source_triangle)

                # Face 2
                for destinationface in destinationfaces:
                    destinationlandmarks = video_process.landmarks2
                    destinationlandmarks_points = []
                    for n in range(0, 68):
                        x = destinationlandmarks.part(n).x
                        y = destinationlandmarks.part(n).y
                        destinationlandmarks_points.append((x, y))


                    destination_triangle_points = np.array(destinationlandmarks_points, np.int32)
                    destinationconvexhull = cv2.convexHull(destination_triangle_points)
                    cv2.fillConvexPoly(destinationmask, destinationconvexhull, 255)  


                # Creating empty mask
                source_lines_space_mask = np.zeros_like(sourcegray)
                destination_lines_space_mask = np.zeros_like(destinationgray)



                # Iterating through all source delaunay triangle and superimposing source triangles in empty destination canvas after warping to same size as destination triangles' shape
                for triangle_index in indexes_triangles:
                    # Triangulation of the first face
                    tr1_pt1 = sourcelandmarks_points[triangle_index[0]]
                    tr1_pt2 = sourcelandmarks_points[triangle_index[1]]
                    tr1_pt3 = sourcelandmarks_points[triangle_index[2]]
                    source_triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


                    source_rectangle = cv2.boundingRect(source_triangle)
                    (x, y, w, h) = source_rectangle
                    (xu, yu, wu, hu) = source_rectangle
                    cropped_source_rectangle = sourceframe[y: yu + hu, x: xu + wu]
                    cropped_source_rectangle_mask = np.zeros((h, w), np.uint8)


                    source_triangle_points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                    [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                    [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_source_rectangle_mask, source_triangle_points, 255)

                    # Lines space
                    cv2.line(source_lines_space_mask, tr1_pt1, tr1_pt2, 255)
                    cv2.line(source_lines_space_mask, tr1_pt2, tr1_pt3, 255)
                    cv2.line(source_lines_space_mask, tr1_pt1, tr1_pt3, 255)

                    # Triangulation of second face
                    tr2_pt1 = destinationlandmarks_points[triangle_index[0]]
                    tr2_pt2 = destinationlandmarks_points[triangle_index[1]]
                    tr2_pt3 = destinationlandmarks_points[triangle_index[2]]
                    destination_triangle = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


                    destination_rectangle = cv2.boundingRect(destination_triangle)
                    (x, y, w, h) = destination_rectangle
                    cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)

        
                    cropped_triangle_source = sourceframe[y: y + h, x: x + w] #OR dest frame?
                    cropped_tr_mask_source = np.zeros((h, w), np.uint8)  #MASK NOT CORRECT YET
    


                    destination_triangle_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)


                    # Lines space vol 2
                    cv2.line(destination_lines_space_mask, tr2_pt1, tr2_pt2, 255)
                    cv2.line(destination_lines_space_mask, tr2_pt2, tr2_pt3, 255)
                    cv2.line(destination_lines_space_mask, tr2_pt1, tr2_pt3, 255)


                    # Warp source triangle to match shape of destination triangle and put it over destination triangle mask
                    source_triangle_points = np.float32(source_triangle_points)
                    destination_triangle_points = np.float32(destination_triangle_points)
                    matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
                    # M2 = cv2.getAffineTransform(destination_triangle_points, source_triangle_points)

                    warped_rectangle = cv2.warpAffine(cropped_source_rectangle, matrix, (w, h))
                    warped_triangle = cv2.bitwise_and(warped_rectangle, warped_rectangle, mask=cropped_destination_rectangle_mask)


                    warped_triangle_2 = cv2.warpAffine(cropped_triangle_source, matrix, (w, h)) #THESE NOT CORRECT YET: CROPPED, M, w, h
                    warped_triangle_2 = cv2.bitwise_and(warped_triangle_2, warped_triangle_2, mask=cropped_tr_mask_source)

                
                    #  Reconstructing destination face in empty canvas of destination image
                    new_dest_face_canvas_area = destination_image_canvas[y: y + h, x: x + w]
                    new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)

                    _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_created_triangle)

                    new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, warped_triangle)
                    destination_image_canvas[y: y + h, x: x + w] = new_dest_face_canvas_area

                    # Reconstructing source face in empty canvas of source image

                    new_source_face_canvas_area = source_image_canvas[y: y + h, x: x + w]    #H, W ETC need editing. #THESE NOT CORRECT YET
                    new_source_face_canvas_area_gray = cv2.cvtColor(new_source_face_canvas_area, cv2.COLOR_BGR2GRAY)
                    _, mask_created_triangle_2 = cv2.threshold(new_source_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle_2 = cv2.bitwise_and(warped_triangle_2, warped_triangle_2, mask=mask_created_triangle_2)

                    new_source_face_canvas_area = cv2.add(new_source_face_canvas_area, warped_triangle_2)
                    source_image_canvas[y: y + h, x: x + w] = new_source_face_canvas_area


                ## Put reconstructed face on the destination image

                opacity = len(pastframes1) * 8 #max value is 24

                final_destination_canvas = np.zeros_like(destinationgray)
                final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destinationconvexhull, opacity) 
                final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)
                destination_face_masked = cv2.bitwise_and(destinationframe, destinationframe, mask=final_destination_canvas)
                result = cv2.add(destination_face_masked, destination_image_canvas)
                

                # Put reconstructed face on the source image

                final_source_canvas = np.zeros_like(sourcegray)
                final_source_face_mask = cv2.fillConvexPoly(final_source_canvas, sourceconvexhull, opacity) #EDITED
                final_source_canvas = cv2.bitwise_not(final_source_face_mask)
                source_face_masked = cv2.bitwise_and(sourceframe, sourceframe, mask=final_source_canvas)
                result2 = cv2.add(source_face_masked , source_image_canvas) 
    

                # Creating seamless clone of two faces
                (x, y, w, h) = cv2.boundingRect(destinationconvexhull)
                center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                (x2, y2, w2, h2) = cv2.boundingRect(sourceconvexhull)
                center_face_source = (int((x2 + x2 + w2) / 2), int((y2 + y2 + h2) / 2))
                seamlessclone = cv2.seamlessClone(result, destinationframe, final_destination_face_mask, center_face2, cv2.NORMAL_CLONE)
                seamlessclone = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)

                seamlessclone2 = cv2.seamlessClone(result2, sourceframe, final_source_face_mask, center_face_source, cv2.NORMAL_CLONE)  #EDITED
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

