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

    video_capture2 = VideoGet(src=0).start()  #1 and 0 at home, 0 and 2 at uni
    video_capture = VideoGet(src=2).start()

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



        
                    person1_currentframe = video_process.frame
                    person1_olderframe = random.choice(pastframes1)

                    heightp1curr, widtp1curr, channelsp1curr = person1_currentframe.shape
                    heightp1old, widtp1old, channelsp1old = person1_olderframe.shape
                
                    person1_currentframe_gray = cv2.cvtColor(person1_currentframe, cv2.COLOR_BGR2GRAY)
                    person1_olderframe_gray = cv2.cvtColor(person1_olderframe, cv2.COLOR_BGR2GRAY)

                    person1currmask = np.zeros_like(person1_currentframe_gray)
                    person1oldmask = np.zeros_like(person1_olderframe_gray)

                    person1curr_image_canvas = np.zeros((heightp1curr, widtp1curr, channelsp1curr), np.uint8)
                    person1old_image_canvas = np.zeros((heightp1old, widtp1old, channelsp1old), np.uint8)

                    indexes_triangles_person1morph = []

                    person1curr_faces = video_process.faces #I'd like to apply this detecyion to consequtive frames 
                    person1old_faces = video_process.faces #from feed 1: maybe need to do the detection using person1_currentframe_gray etc here

                    for person1curr_face in person1curr_faces:
                        sourcelandmarksp1c = video_process.landmarks
                        sourcelandmarks_pointsp1c = []
                        for n in range(0, 68):
                            x = sourcelandmarksp1c.part(n).x
                            y = sourcelandmarksp1c.part(n).y
                            sourcelandmarks_pointsp1c.append((x, y))

                        p1c_triangle_points = np.array(sourcelandmarks_pointsp1c, np.int32)
                        p1cconvexhull = cv2.convexHull(p1c_triangle_points)
                        cv2.fillConvexPoly(person1currmask, p1cconvexhull, 255)

                        # Delaunay triangulation

                        p1crect = cv2.boundingRect(p1cconvexhull)
                        p1csubdiv = cv2.Subdiv2D(p1crect)
                        p1csubdiv.insert(sourcelandmarks_pointsp1c)
                        p1ctriangles = p1csubdiv.getTriangleList()
                        p1ctriangles = np.array(p1ctriangles, dtype=np.int32)

                        for t in p1ctriangles:
                            pt1 = (t[0], t[1])
                            pt2 = (t[2], t[3])
                            pt3 = (t[4], t[5])

                            index_pt1 = np.where((p1c_triangle_points == pt1).all(axis=1))
                            index_pt1 = extract_index_nparray(index_pt1)

                            index_pt2 = np.where((p1c_triangle_points == pt2).all(axis=1))
                            index_pt2 = extract_index_nparray(index_pt2)

                            index_pt3 = np.where((p1c_triangle_points == pt3).all(axis=1))
                            index_pt3 = extract_index_nparray(index_pt3)
                            
                            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                                p1c_triangle = [index_pt1, index_pt2, index_pt3]
                                indexes_triangles_person1morph.append(p1c_triangle)

                    # Face 2
                    for person1oldface in person1old_faces:
                        destinationlandmarksp1o = video_process.landmarks
                        destinationlandmarks_pointsp1o = []
                        for n in range(0, 68):
                            x = destinationlandmarksp1o.part(n).x
                            y = destinationlandmarksp1o.part(n).y
                            destinationlandmarks_pointsp1o.append((x, y))

                        p1o_triangle_points = np.array(destinationlandmarks_pointsp1o, np.int32)
                        p1oconvexhull = cv2.convexHull(p1o_triangle_points)
                        cv2.fillConvexPoly(person1oldmask, p1oconvexhull, 255)  


                    # Iterating through all source delaunay triangle and superimposing source triangles in empty destination canvas after warping to same size as destination triangles' shape
                    for triangle_indexp1 in indexes_triangles_person1morph:

                        # Triangulation of the first face
                        tr1_pt1 = sourcelandmarks_pointsp1c[triangle_indexp1[0]]
                        tr1_pt2 = sourcelandmarks_pointsp1c[triangle_indexp1[1]]
                        tr1_pt3 = sourcelandmarks_pointsp1c[triangle_indexp1[2]]
                        source_trianglep1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                        # Source rectangle
                        source_rectanglep1 = cv2.boundingRect(source_trianglep1)
                        (xc, yc, wc, hc) = source_rectanglep1

                        cropped_source_rectanglep1 = person1_currentframe[yc: yc + hc, xc: xc + wc] #even destinationframe here returns empty. tested yu and xu
                        cropped_source_rectangle_maskp1 = np.zeros((hc, wc), np.uint8)

                        p1c_triangle_points = np.array([[tr1_pt1[0] - xc, tr1_pt1[1] - yc],
                                        [tr1_pt2[0] - xc, tr1_pt2[1] - yc],
                                        [tr1_pt3[0] - xc, tr1_pt3[1] - yc]], np.int32) # should be xu, was x y etc beofre

                        cv2.fillConvexPoly(cropped_source_rectangle_maskp1, p1c_triangle_points, 255)


                        # Triangulation of second face
                        tr2_pt1 = destinationlandmarks_pointsp1o[triangle_indexp1[0]]
                        tr2_pt2 = destinationlandmarks_pointsp1o[triangle_indexp1[1]]
                        tr2_pt3 = destinationlandmarks_pointsp1o[triangle_indexp1[2]]
                        destination_trianglep1 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


                        # Dest rectangle, WORKS
                        destination_rectanglep1 = cv2.boundingRect(destination_trianglep1)
                        (xo, yo, wo, ho) = destination_rectanglep1

                        cropped_destination_rectanglep1 = person1_olderframe[yo: yo + ho, xo: xo + wo]  #was sourceframe and worked
                        cropped_destination_rectangle_maskp1 = np.zeros((ho, wo), np.uint8)

                        p1o_triangle_points = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                        cv2.fillConvexPoly(cropped_destination_rectangle_maskp1, p1o_triangle_points, 255)


                        # Warp source triangle to match shape of destination triangle and put it over destination triangle mask
                        p1c_triangle_points = np.float32(p1c_triangle_points)
                        p1o_triangle_points = np.float32(p1o_triangle_points)

                        matrixp1 = cv2.getAffineTransform(p1c_triangle_points, p1o_triangle_points)
                    
                        warped_rectanglep1 = cv2.warpAffine(cropped_source_rectanglep1 , matrixp1, (wo, ho))
                        warped_trianglep1 = cv2.bitwise_and(warped_rectanglep1, warped_rectanglep1, mask=cropped_destination_rectangle_maskp1)

                        #  Reconstructing destination face in empty canvas of destination image
                        new_dest_face_canvas_areap1 = person1old_image_canvas[yo: yo + ho, xo: xo + wo] # h y etc. are from dest rect and it works
                        new_dest_face_canvas_areap1_gray= cv2.cvtColor(new_dest_face_canvas_areap1, cv2.COLOR_BGR2GRAY)

                        _, mask_created_trianglep1 = cv2.threshold(new_dest_face_canvas_areap1_gray, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_trianglep1 = cv2.bitwise_and(warped_trianglep1, warped_trianglep1, mask=mask_created_trianglep1)

                        new_dest_face_canvas_areap1 = cv2.add(new_dest_face_canvas_areap1, warped_trianglep1)
                        person1old_image_canvas[yo: yo + ho, xo: xo + wo] = new_dest_face_canvas_areap1
                    

                    ## Put reconstructed face on the destination image

                    #opacity = len(pastframes1) * 8 #max value is 24

                    final_destination_canvasp1 = np.zeros_like(person1_olderframe_gray)
                    final_destination_face_maskp1 = cv2.fillConvexPoly(final_destination_canvasp1, p1oconvexhull, 155) 
                    final_destination_canvasp1 = cv2.bitwise_not(final_destination_face_maskp1)
                    destination_face_maskedp1 = cv2.bitwise_and(person1_olderframe, person1_olderframe, mask=final_destination_canvasp1)
                    result3 = cv2.add(destination_face_maskedp1, person1old_image_canvas)
                    

                    # Creating seamless clone of two faces
                    (x, y, w, h) = cv2.boundingRect(p1oconvexhull)
                    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                    (x2, y2, w2, h2) = cv2.boundingRect(p1cconvexhull)
                    center_face_source = (int((x2 + x2 + w2) / 2), int((y2 + y2 + h2) / 2))
                    seamlessclone3 = cv2.seamlessClone(result3, person1_olderframe, final_destination_face_maskp1, center_face2, cv2.NORMAL_CLONE)
                    seamlessclone3 = cv2.cvtColor(seamlessclone3, cv2.COLOR_BGR2GRAY)

                    resultframe2 = seamlessclone3
                    resultframe = seamlessclone3
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
                    cv2.polylines(resultframe2, [video_process.lefteye_start, video_process.lefteye_end], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

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


