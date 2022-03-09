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
# potentially add ml to generate nww faces


import cv2
import numpy as np
import dlib
from PIL import Image
import math
from VideoGet import VideoGet

# Helper function for extracting index from array
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def threadVideoget(source=0):
    video_getter = VideoGet(source).start()
    #cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break
        frame = video_getter.frame
        #frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        #cps.increment()


#for handling old sequences
pastframes1 = list()
pastframes2 = list()

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

threadVideoget(0) #inits while loop for getting frames from camera 0
threadVideoget(1) #inits while loop for getting frames from camera 1

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
    ret3, frame3= cap2.read()#placeholder, will show video from the room
    ret4, frame4= cap2.read()#placeholder, will show video from the room

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

    for face2 in faces2:
        faceleft2 = face2.left()
        facetop2 = face2.top()
        faceright2 = face2.right()
        facebottom2 = face2.bottom()
        cv2.rectangle(frame, (faceleft2, facetop2), (faceright2, facebottom2), (0, 255, 0), 3)

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

        #delanay effect neing used here

        mask = np.zeros_like(gray)
        height, width, channels = frame2.shape
        img2_new_face = np.zeros((height, width, channels), np.uint8)
        indexes_triangles = []


        mask2 = np.zeros_like(gray2)
        height2, width2, channels2 = frame.shape
        img1_new_face = np.zeros((height2, width2, channels2), np.uint8)
        indexes_triangles2 = []

        #landmarks of first face

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

            points = np.array(landmarks_points, np.int32)

            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, convexhull, 255)    
            
            face_image_1 = cv2.bitwise_and(frame, frame, mask=mask)
     

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
            landmarks = predictor(gray2, face)
            landmarks_points2 = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points2.append((x, y))


            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)
            cv2.fillConvexPoly(mask2, convexhull2, 255)  
            face_image_2 = cv2.bitwise_and(frame2, frame2, mask=mask2)


        # Creating empty mask
        lines_space_mask = np.zeros_like(gray)
        lines_space_new_face = np.zeros_like(frame2)

        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = frame[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)


            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Lines space
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
            cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
            lines_space = cv2.bitwise_and(frame, frame, mask=lines_space_mask)

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

        img2_head_noface = cv2.bitwise_and(frame2, frame2, mask=img2_face_mask)
        img2_head_noface2 = cv2.bitwise_and(frame2, frame2, mask=img2_face_mask2)
        result = cv2.add(img2_head_noface, img2_new_face)
        result2 = cv2.add(img2_head_noface2, img2_new_face)

        # Creating seamless clone of two faces
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv2.seamlessClone(result, frame2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
        seamlessclone = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)

        seamlessclone2 = cv2.seamlessClone(result2, frame2, img2_head_mask2, center_face2, cv2.NORMAL_CLONE)
        seamlessclone2 = cv2.cvtColor(seamlessclone2, cv2.COLOR_BGR2GRAY)


        # Converting array to image
        #resultimage = Image.fromarray(seamlessclone)
        #cv2.imshow("result", seamlessclone) #

        #if morphing in progress, frame3 becomes our morph result
        frame3 = seamlessclone
        frame4 = seamlessclone2


    else:
        is_recording = False

    #two outputs for showing effects and feedback    

    inputs = np.concatenate((frame, frame2), axis=0)
    outputs = np.concatenate((frame3, frame4), axis=0)
    print(len(pastframes1))
    print(len(pastframes2))


    #cv2.imshow("Inputs", inputs) #showing input and detection
    cv2.imshow("Result", outputs) #showing input and detection

    key = cv2.waitKey(1)
    if key == 27: #esc
        break

cv2.destroyAllWindows()

