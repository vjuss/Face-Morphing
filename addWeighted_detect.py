#steps to do: 
#4) effet 2
#6) draw nicer version of the eyes
#7) music
#8) test on pi or mac mini
#9) code into classes, fucntions etc


from cgitb import handler
from ctypes import wstring_at
from errno import EALREADY
from locale import currency
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


def translate(value, leftMin, leftMax, rightMin, rightMax): #https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)




def main():

    video_capture2 = VideoGet(src=1).start()  #1 and 0 at home, 0 and 2 at uni
    video_capture = VideoGet(src=0).start()

    facedetector = dlib.get_frontal_face_detector()
    landmarkpredictor = dlib.shape_predictor("data/68_face_landmarks.dat")
    
    video_process= CheckFaceLoc(capture1 = video_capture, capture2=video_capture2, detector=facedetector, predictor=landmarkpredictor).start()
    #FIX THIS SO THAT YOU ONLY USE CERATIN PROPERTIES FROM SAME VIDOE PROCESS CLASS. CREATE FUNCTIONS THERE

    #for handling old sequences
    pastframes1 = list()
    pastframes2 = list()

    matchresult = False # placeholder when starting
    drawingeyes = False

    while True:
        if video_capture.stopped:
                video_capture.stop()
                break

        vidframe = video_capture.frame
        vidframe2 = video_capture2.frame

        twofaces = video_process.twofaces

        current_time = time.time() #keeps updating
       

        if twofaces == True:
            matchresult = video_process.match

            if matchresult == True:
            
                pastframes1.append(vidframe) # storing ghost images to be used later
                pastframes2.append(vidframe2) #
                
                elapsed_time = current_time - start_time
                #print(elapsed_time)
                #
                #

                if elapsed_time < 10:
                    sourceframe = video_process.frame
                    destinationframe = video_process.frame2
                    height2, width2, channels2 = sourceframe.shape
                    height, width, channels = destinationframe.shape #was vidframe2
                    

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


                    # Iterating through all source delaunay triangle and superimposing source triangles in empty destination canvas after warping to same size as destination triangles' shape
                    for triangle_index in indexes_triangles:

                        # Triangulation of the first face
                        tr1_pt1 = sourcelandmarks_points[triangle_index[0]]
                        tr1_pt2 = sourcelandmarks_points[triangle_index[1]]
                        tr1_pt3 = sourcelandmarks_points[triangle_index[2]]
                        source_triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                        # Source rectangle
                        source_rectangle = cv2.boundingRect(source_triangle)
                        (xu, yu, wu, hu) = source_rectangle
                        cropped_source_rectangle = sourceframe[yu: yu + hu, xu: xu + wu] #even destinationframe here returns empty. tested yu and xu
                        cropped_source_rectangle_mask = np.zeros((hu, wu), np.uint8)

                        source_triangle_points = np.array([[tr1_pt1[0] - xu, tr1_pt1[1] - yu],
                                        [tr1_pt2[0] - xu, tr1_pt2[1] - yu],
                                        [tr1_pt3[0] - xu, tr1_pt3[1] - yu]], np.int32) # should be xu, was x y etc beofre

                        cv2.fillConvexPoly(cropped_source_rectangle_mask, source_triangle_points, 255)



                        # Triangulation of second face
                        tr2_pt1 = destinationlandmarks_points[triangle_index[0]]
                        tr2_pt2 = destinationlandmarks_points[triangle_index[1]]
                        tr2_pt3 = destinationlandmarks_points[triangle_index[2]]
                        destination_triangle = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                        # Dest rectangle, WORKS
                        destination_rectangle = cv2.boundingRect(destination_triangle)
                        (x, y, w, h) = destination_rectangle

                        cropped_destination_rectangle = destinationframe[y: y + h, x: x + w]  #was sourceframe and worked
                        cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)

                        destination_triangle_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                        cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)


                        # Warp source triangle to match shape of destination triangle and put it over destination triangle mask
                        source_triangle_points = np.float32(source_triangle_points)
                        destination_triangle_points = np.float32(destination_triangle_points)

                        matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
                        matrix2 = cv2.getAffineTransform(destination_triangle_points, source_triangle_points)

                        warped_rectangle = cv2.warpAffine(cropped_source_rectangle, matrix, (w, h))
                        warped_triangle = cv2.bitwise_and(warped_rectangle, warped_rectangle, mask=cropped_destination_rectangle_mask)

                        warped_rectangle_2 = cv2.warpAffine(cropped_destination_rectangle, matrix2, (wu, hu)) 
                        warped_triangle_2 = cv2.bitwise_and(warped_rectangle_2, warped_rectangle_2, mask=cropped_source_rectangle_mask)

                    
                        #  Reconstructing destination face in empty canvas of destination image
                    
                        new_dest_face_canvas_area = destination_image_canvas[y: y + h, x: x + w] # h y etc. are from dest rect and it works
                        new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)

                        _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_created_triangle)

                        new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, warped_triangle)
                        destination_image_canvas[y: y + h, x: x + w] = new_dest_face_canvas_area

                        # Reconstructing source face in empty canvas of source image
                        new_source_face_canvas_area = source_image_canvas[yu: yu + hu, xu: xu + wu]    # THIS WAS EMPTY when y: yu = hu
                        new_source_face_canvas_area_gray = cv2.cvtColor(new_source_face_canvas_area, cv2.COLOR_BGR2GRAY)
                        _, mask_created_triangle_2 = cv2.threshold(new_source_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_triangle_2 = cv2.bitwise_and(warped_triangle_2, warped_triangle_2, mask=mask_created_triangle_2)

                        new_source_face_canvas_area = cv2.add(new_source_face_canvas_area, warped_triangle_2)
                        source_image_canvas[yu: yu + hu, xu: xu + wu] = new_source_face_canvas_area


                    ## Put reconstructed face on the destination image

                    opacity = translate(elapsed_time, 0, 10, 0, 255)
                    print("map wth function", opacity)
    
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

                    resultframe = seamlessclone2
                    resultframe2 = seamlessclone
                #
                #
                #
                #
                #
            

                elif elapsed_time >= 10 and elapsed_time < 20:
                    print("effect 2")

                    #for both participants, delaunay becomes time travel between their own frames. Testing with one first

                    #
                    #
                    #
                    person1_sourceframe = random.choice(pastframes2) #old frame is the source
                    person1_destframe = video_process.frame2 #curetn frme is the destination
                    

                    height2p, width2p, channels2p = person1_sourceframe.shape
                    heightp, widthp, channelsp = person1_destframe.shape #was vidframe2

                    sourcegray_person1 = cv2.cvtColor(person1_sourceframe, cv2.COLOR_BGR2GRAY) 
                    destinationgray_person1 = cv2.cvtColor(person1_destframe, cv2.COLOR_BGR2GRAY) 

                    sourcemaskp = np.zeros_like(sourcegray_person1)
                    destinationmaskp = np.zeros_like(destinationgray_person1)

                    source_image_canvasp = np.zeros((height2p, width2p, channels2p), np.uint8)
                    destination_image_canvasp = np.zeros((heightp, widthp, channelsp), np.uint8)

                    indexes_trianglesp = []

                    #landmarks of the past face

                    sourcefacesp = facedetector(person1_sourceframe)
                    destinationfacesp = video_process.faces #current faces



                    for sourcefacep in sourcefacesp:
                        sourcelandmarksp = landmarkpredictor(sourcegray_person1, sourcefacep) #NEED TO GET PAST FSCES HERE, FIX 
                        sourcelandmarks_pointsp = []
                        for n in range(0, 68):
                            x = sourcelandmarksp.part(n).x
                            y = sourcelandmarksp.part(n).y
                            sourcelandmarks_pointsp.append((x, y))

                        source_triangle_pointsp = np.array(sourcelandmarks_pointsp, np.int32)
                        sourceconvexhullp = cv2.convexHull(source_triangle_pointsp)
                        cv2.fillConvexPoly(sourcemaskp, sourceconvexhullp, 255)    
                        

                        # Delaunay triangulation

                        sourcerectp = cv2.boundingRect(sourceconvexhullp)
                        sourcesubdivp = cv2.Subdiv2D(sourcerectp)
                        sourcesubdivp.insert(sourcelandmarks_pointsp)
                        sourcetrianglesp = sourcesubdivp.getTriangleList()
                        sourcetrianglesp = np.array(sourcetrianglesp, dtype=np.int32)
                        

                        for t in sourcetrianglesp:
                            pt1p = (t[0], t[1])
                            pt2p = (t[2], t[3])
                            pt3p = (t[4], t[5])

                            index_pt1p = np.where((source_triangle_pointsp == pt1p).all(axis=1))
                            index_pt1p = extract_index_nparray(index_pt1p)

                            index_pt2p = np.where((source_triangle_pointsp == pt2p).all(axis=1))
                            index_pt2p = extract_index_nparray(index_pt2p)

                            index_pt3p = np.where((source_triangle_pointsp == pt3p).all(axis=1))
                            index_pt3p = extract_index_nparray(index_pt3p)
                            
                            if index_pt1p is not None and index_pt2p is not None and index_pt3p is not None:
                                source_trianglep = [index_pt1p, index_pt2p, index_pt3p]
                                indexes_trianglesp.append(source_trianglep)


                    # Face 2 (current face)
                    for destinationfacep in destinationfacesp:
                        destinationlandmarksp = video_process.landmarks
                        destinationlandmarks_pointsp = []
                        for n in range(0, 68):
                            x = destinationlandmarksp.part(n).x
                            y = destinationlandmarksp.part(n).y
                            destinationlandmarks_pointsp.append((x, y))

                        destination_triangle_pointsp = np.array(destinationlandmarks_pointsp, np.int32)
                        destinationconvexhullp = cv2.convexHull(destination_triangle_pointsp)
                        cv2.fillConvexPoly(destinationmaskp, destinationconvexhullp, 255)  




                    # Iterating through all source delaunay triangle and superimposing source triangles in empty destination canvas after warping to same size as destination triangles' shape
                    for triangle_indexp in indexes_trianglesp:

                        # Triangulation of the first face
                        tr1_pt1p = sourcelandmarks_pointsp[triangle_indexp[0]]
                        tr1_pt2p = sourcelandmarks_pointsp[triangle_indexp[1]]
                        tr1_pt3p = sourcelandmarks_pointsp[triangle_indexp[2]]
                        source_trianglep = np.array([tr1_pt1p, tr1_pt2p, tr1_pt3p], np.int32)

                        # Source rectangle
                        source_rectanglep = cv2.boundingRect(source_trianglep)
                        (xup, yup, wup, hup) = source_rectanglep
                        cropped_source_rectanglep = person1_sourceframe[yup: yup + hup, xup: xup + wup] #even destinationframe here returns empty. tested yu and xu
                        cropped_source_rectangle_maskp = np.zeros((hup, wup), np.uint8)

                        source_triangle_pointsp = np.array([[tr1_pt1p[0] - xup, tr1_pt1p[1] - yup],
                                        [tr1_pt2p[0] - xup, tr1_pt2p[1] - yup],
                                        [tr1_pt3p[0] - xup, tr1_pt3p[1] - yup]], np.int32) # should be xu, was x y etc beofre

                        cv2.fillConvexPoly(cropped_source_rectangle_maskp, source_triangle_pointsp, 255)


                        # Triangulation of second face
                        tr2_pt1p = destinationlandmarks_pointsp[triangle_indexp[0]]
                        tr2_pt2p = destinationlandmarks_pointsp[triangle_indexp[1]]
                        tr2_pt3p = destinationlandmarks_pointsp[triangle_indexp[2]]
                        destination_trianglep = np.array([tr2_pt1p, tr2_pt2p, tr2_pt3p], np.int32)

                        # Dest rectangle, WORKS
                        destination_rectanglep = cv2.boundingRect(destination_trianglep)
                        (xp, yp, wp, hp) = destination_rectanglep

                        cropped_destination_rectanglep = person1_destframe[yp: yp + hp, xp: xp + wp]  #was sourceframe and worked
                        cropped_destination_rectangle_maskp = np.zeros((hp, wp), np.uint8)

                        destination_triangle_pointsp = np.array([[tr2_pt1p[0] - xp, tr2_pt1p[1] - yp],
                                            [tr2_pt2p[0] - xp, tr2_pt2p[1] - yp],
                                            [tr2_pt3p[0] - xp, tr2_pt3p[1] - yp]], np.int32)

                        cv2.fillConvexPoly(cropped_destination_rectangle_maskp, destination_triangle_pointsp, 255)


                        # Warp source triangle to match shape of destination triangle and put it over destination triangle mask
                        source_triangle_pointsp = np.float32(source_triangle_pointsp)
                        destination_triangle_pointsp = np.float32(destination_triangle_pointsp)

                        matrixp = cv2.getAffineTransform(source_triangle_pointsp, destination_triangle_pointsp)
                        matrix2p = cv2.getAffineTransform(destination_triangle_pointsp, source_triangle_pointsp)

                        warped_rectanglep = cv2.warpAffine(cropped_source_rectanglep, matrixp, (wp, hp))
                        warped_trianglep = cv2.bitwise_and(warped_rectanglep, warped_rectanglep, mask=cropped_destination_rectangle_maskp)

                         #warped_rectangle_2 = cv2.warpAffine(cropped_destination_rectangle, matrix2, (wu, hu)) 
                         #warped_triangle_2 = cv2.bitwise_and(warped_rectangle_2, warped_rectangle_2, mask=cropped_source_rectangle_mask)

                    
                        #  Reconstructing destination face in empty canvas of destination image
                    
                        new_dest_face_canvas_areap = destination_image_canvasp[yp: yp + hp, xp: xp + wp] # h y etc. are from dest rect and it works
                        new_dest_face_canvas_area_grayp = cv2.cvtColor(new_dest_face_canvas_areap, cv2.COLOR_BGR2GRAY)

                        _, mask_created_trianglep = cv2.threshold(new_dest_face_canvas_area_grayp, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_trianglep = cv2.bitwise_and(warped_trianglep, warped_trianglep, mask=mask_created_trianglep)

                        new_dest_face_canvas_areap = cv2.add(new_dest_face_canvas_areap, warped_trianglep)
                        destination_image_canvasp[yp: yp + hp, xp: xp + wp] = new_dest_face_canvas_areap


                     ## Put reconstructed face on the destination image

                    opacity = translate(elapsed_time, 10, 20, 255, 0)
                    print("map wth function", opacity)
    
                    final_destination_canvasp = np.zeros_like(destinationgray_person1)
                    final_destination_face_maskp = cv2.fillConvexPoly(final_destination_canvasp, destinationconvexhullp, 255) 
                    final_destination_canvasp = cv2.bitwise_not(final_destination_face_maskp)
                    destination_face_maskedp = cv2.bitwise_and(person1_destframe, person1_destframe, mask=final_destination_canvasp)
                    result = cv2.add(destination_face_maskedp, destination_image_canvasp)
                    

                    # Creating seamless clone of two faces
                    (x, y, w, h) = cv2.boundingRect(destinationconvexhullp)
                    center_face2p = (int((x + x + w) / 2), int((y + y + h) / 2))

                    seamlessclone = cv2.seamlessClone(result, person1_destframe, final_destination_face_maskp, center_face2p, cv2.NORMAL_CLONE)
                    seamlessclone = cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2GRAY)

                    resultframe = seamlessclone
                    resultframe2 = seamlessclone
    
                #
                #
                #
                #
                #
                elif elapsed_time >= 20 and elapsed_time < 30:
                #
                #
                #
                    print("effect 3")
                    sourceframe = video_process.frame
                    destinationframe = video_process.frame2
                    height2, width2, channels2 = sourceframe.shape
                    height, width, channels = destinationframe.shape #was vidframe2
                    

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


                    # Iterating through all source delaunay triangle and superimposing source triangles in empty destination canvas after warping to same size as destination triangles' shape
                    for triangle_index in indexes_triangles:

                        # Triangulation of the first face
                        tr1_pt1 = sourcelandmarks_points[triangle_index[0]]
                        tr1_pt2 = sourcelandmarks_points[triangle_index[1]]
                        tr1_pt3 = sourcelandmarks_points[triangle_index[2]]
                        source_triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                        # Source rectangle
                        source_rectangle = cv2.boundingRect(source_triangle)
                        (xu, yu, wu, hu) = source_rectangle
                        cropped_source_rectangle = sourceframe[yu: yu + hu, xu: xu + wu] #even destinationframe here returns empty. tested yu and xu
                        cropped_source_rectangle_mask = np.zeros((hu, wu), np.uint8)

                        source_triangle_points = np.array([[tr1_pt1[0] - xu, tr1_pt1[1] - yu],
                                        [tr1_pt2[0] - xu, tr1_pt2[1] - yu],
                                        [tr1_pt3[0] - xu, tr1_pt3[1] - yu]], np.int32) # should be xu, was x y etc beofre

                        cv2.fillConvexPoly(cropped_source_rectangle_mask, source_triangle_points, 255)



                        # Triangulation of second face
                        tr2_pt1 = destinationlandmarks_points[triangle_index[0]]
                        tr2_pt2 = destinationlandmarks_points[triangle_index[1]]
                        tr2_pt3 = destinationlandmarks_points[triangle_index[2]]
                        destination_triangle = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                        # Dest rectangle, WORKS
                        destination_rectangle = cv2.boundingRect(destination_triangle)
                        (x, y, w, h) = destination_rectangle

                        cropped_destination_rectangle = destinationframe[y: y + h, x: x + w]  #was sourceframe and worked
                        cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)

                        destination_triangle_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                        cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)


                        # Warp source triangle to match shape of destination triangle and put it over destination triangle mask
                        source_triangle_points = np.float32(source_triangle_points)
                        destination_triangle_points = np.float32(destination_triangle_points)

                        matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
                        matrix2 = cv2.getAffineTransform(destination_triangle_points, source_triangle_points)

                        warped_rectangle = cv2.warpAffine(cropped_source_rectangle, matrix, (w, h))
                        warped_triangle = cv2.bitwise_and(warped_rectangle, warped_rectangle, mask=cropped_destination_rectangle_mask)

                        warped_rectangle_2 = cv2.warpAffine(cropped_destination_rectangle, matrix2, (wu, hu)) 
                        warped_triangle_2 = cv2.bitwise_and(warped_rectangle_2, warped_rectangle_2, mask=cropped_source_rectangle_mask)

                    
                        #  Reconstructing destination face in empty canvas of destination image
                    
                        new_dest_face_canvas_area = destination_image_canvas[y: y + h, x: x + w] # h y etc. are from dest rect and it works
                        new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)

                        _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_created_triangle)

                        new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, warped_triangle)
                        destination_image_canvas[y: y + h, x: x + w] = new_dest_face_canvas_area

                        # Reconstructing source face in empty canvas of source image
                        new_source_face_canvas_area = source_image_canvas[yu: yu + hu, xu: xu + wu]    # THIS WAS EMPTY when y: yu = hu
                        new_source_face_canvas_area_gray = cv2.cvtColor(new_source_face_canvas_area, cv2.COLOR_BGR2GRAY)
                        _, mask_created_triangle_2 = cv2.threshold(new_source_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_triangle_2 = cv2.bitwise_and(warped_triangle_2, warped_triangle_2, mask=mask_created_triangle_2)

                        new_source_face_canvas_area = cv2.add(new_source_face_canvas_area, warped_triangle_2)
                        source_image_canvas[yu: yu + hu, xu: xu + wu] = new_source_face_canvas_area


                    ## Put reconstructed face on the destination image

                    opacity = translate(elapsed_time, 20, 30, 255, 0)
                    print("map wth function", opacity)
    
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

                    resultframe = seamlessclone2
                    resultframe2 = seamlessclone

                else:
                    print("timer full, reset the sketch")
                    resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                    resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY)

                #THESE 3 LINES ARE THE GOAL, NOT WORKING YET
                #delaunay_process = AddDelaunay(frame = vidframe, frame2 = vidframe2, detector=facedetector, predictor = landmarkpredictor).start()
                #resultframe = delaunay_process.seamlessclone
                #resultframe2 = delaunay_process.seamlessclone2
            #
            #
            #

            else: # if match result not true but there are two faces, just draw eyes and update start time
                start_time = time.time() 
                resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

                if len(video_process.rightpupils) ==2 and len(video_process.rightpupils2) ==2:
                    cv2.circle(resultframe2, video_process.rightpupils, 20, (0), -1) #drawing eyes to opponent's frame
                    cv2.circle(resultframe2, video_process.leftpupils, 20, (0), -1)
                    cv2.circle(resultframe, video_process.rightpupils2, 20, (255), -1)
                    cv2.circle(resultframe, video_process.leftpupils2, 20, (255), -1)
                    #cv2.polylines(resultframe2, [video_process.lefteye_start, video_process.lefteye_end], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

                else:
                    drawingeyes = False

        else: # if no two faces
            drawingeyes = False
            resultframe = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 
            resultframe2 = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 

        # this happens in any case, we just manipulate the contents
        #
        #

        cv2.imshow("Veerasview", resultframe)
        cv2.imshow("Otherview", resultframe2)

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


