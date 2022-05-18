#steps to do: 
#FOR FUTURE VERSION AFTER CORSICA: maybe still try face blending between each persons current and past? when code more simple with classes
#FOR FUTURE VERSION AFTER CORSICA: https://www.makeartwithpython.com/blog/building-a-snapchat-lens-effect-in-python/
#IF TIME: full screeen mode
#MUST DO : TEST MORPH AMOUNTS,



from cgitb import handler
from ctypes import wstring_at
from errno import EALREADY
from locale import currency
import cv2
from cv2 import FLOODFILL_FIXED_RANGE
import numpy as np
import dlib
import math

from pyparsing import null_debug_action
from VideoGet import VideoGet
from CheckFaceLoc import CheckFaceLoc
import random
import threading
import time
import argparse
from pythonosc import udp_client


class EyeList(object):
    def __init__(self, length):
        self.length = length
        self.eyes = []

    def push(self, newcoords):
        if len(self.eyes) < self.length:
            self.eyes.append(newcoords)
        else:
            self.eyes.pop(0)
            self.eyes.append(newcoords)
    
    def clear(self):
        self.eyes = []

facelist = EyeList(10)



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


def makeDelaunay(srcframe, destframe, srcfaces, destfaces, srclandmarks, destlandmarks, elapsedtime, mintime, maxtime, startamount, endamount):
    sourceframe = srcframe
    destinationframe = destframe
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

    sourcefaces = srcfaces
    destinationfaces = destfaces

    for sourceface in sourcefaces:
        sourcelandmarks = srclandmarks
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
        destinationlandmarks = destlandmarks
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
        # plainmask = cv2.bitwise_and(new_source_face_canvas_area_gray, new_source_face_canvas_area_gray, mask=mask_created_triangle_2)
        # plainx, plainy, plainw, plainh = cv2.boundingRect(plainmask)
        # facelist.push([plainx, plainy])

    

        new_source_face_canvas_area = cv2.add(new_source_face_canvas_area, warped_triangle_2)
        source_image_canvas[yu: yu + hu, xu: xu + wu] = new_source_face_canvas_area


    ## Put reconstructed face on the destination image

    elapsed_time = elapsedtime
    min_time = mintime
    max_time = maxtime
    start_amount = startamount
    end_amount = endamount
    opacity = translate(elapsed_time, min_time, max_time, start_amount, end_amount) ##were 0, 10, 0, 255

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

    _resultframe = seamlessclone2
    _resultframe2 = seamlessclone


    return _resultframe, _resultframe2


def rescale_frame(frame, percent):
    width = int(frame.shape[1]*percent/100)
    height = int(frame.shape[0]*percent/100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main():

    screen_width = 2560
    screen_height = 1440

    cv2.namedWindow("Person1", cv2.WINDOW_NORMAL) #makes it scalable
    cv2.namedWindow("Person2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Person1", 200, 200)
    #cv2.setWindowProperty("Person1", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.setWindowProperty("Person2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    video_capture2 = VideoGet(src=1).start()  #1 and 0 at home, 0 and 2 at uni. 0  and 1 with on own laptop at uni
    video_capture = VideoGet(src=0).start()

    facedetector = dlib.get_frontal_face_detector()
    landmarkpredictor = dlib.shape_predictor("data/68_face_landmarks.dat")
    
    video_process= CheckFaceLoc(capture1 = video_capture, capture2=video_capture2, detector=facedetector, predictor=landmarkpredictor).start()
    #video_process= CheckFaceLoc(capture1 = video_capture.frame, capture2=video_capture2.frame, detector=facedetector, predictor=landmarkpredictor).start()
    #FIX THIS SO THAT YOU ONLY USE CERATIN PROPERTIES FROM SAME VIDOE PROCESS CLASS. CREATE FUNCTIONS THERE

    #for handling old sequences
    pastframes1 = list()
    pastframes2 = list()

    matchresult = False # placeholder when starting
    drawingeyes = False

    #OSC BITS HERE

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.0.2",
      help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5005,
      help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)


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

                client.send_message("/filter", 1) #inform max patch that connection is established
            
                pastframes1.append(vidframe) # storing ghost images to be used later
                pastframes2.append(vidframe2) #
                
                elapsed_time = current_time - start_time

                if elapsed_time < 10:
                    results = makeDelaunay(video_process.frame, video_process.frame2, video_process.faces, video_process.faces2, video_process.landmarks, video_process.landmarks2, elapsed_time, 0, 10, 0, 150) #amount was 0, 200 for long
                    resultframe = results[0]
                    resultframe2 = results[1]
                    cv2.imshow("Person1", resultframe)
                    cv2.imshow("Person2", resultframe2)
                    cv2.waitKey(100)
            
                elif elapsed_time >= 10 and elapsed_time < 20:
                    #BASIC OPTION. THIS IS JUST USING PAST FRAMES FROM BOTH AND CREATING GLITCH
                    if int(elapsed_time) % 2 == 0: # if divisible by 2, use past
                        source_frame = random.choice(pastframes1) #WHEN USING PAST FRAME, FACES STILL COME FROM CURRENT FRAMES. HAS BEEN LIKE THIS WITHOUT THE CURRENT FUNCTION ALREADY 
                        destination_frame = random.choice(pastframes2)
                    else: 
                        source_frame = video_process.frame
                        destination_frame = video_process.frame2


                    #NEW STUFF FROM HERE

                    gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(destination_frame, cv2.COLOR_BGR2GRAY)

                    pastfaces= facedetector(gray)
                    pastfaces2= facedetector(gray2)

                    for pastface in pastfaces:
                        pastlandmarks = landmarkpredictor(gray, pastface)

                    for pastface2 in pastfaces2:
                        pastlandmarks2 = landmarkpredictor(gray2, pastface2)

                    #UNTIL HERE
                    
                    #OLD ONE results = makeDelaunay(source_frame, destination_frame, video_process.faces, video_process.faces2, video_process.landmarks, video_process.landmarks2, elapsed_time, 10, 20, 150, 150) #amount was 200, 200 for long
                    results = makeDelaunay(source_frame, destination_frame, pastfaces, pastfaces2, pastlandmarks, pastlandmarks2, elapsed_time, 10, 20, 150, 150) #amount was 200, 200 for long
                    resultframe = results[0]
                    resultframe2 = results[1]
                    cv2.imshow("Person1", resultframe)
                    cv2.imshow("Person2", resultframe2)
                    cv2.waitKey(100)

                elif elapsed_time >= 20 and elapsed_time < 30:

                    results = makeDelaunay(video_process.frame, video_process.frame2, video_process.faces, video_process.faces2, video_process.landmarks, video_process.landmarks2, elapsed_time, 20, 30, 150, 0) #amount was 200, 0 for long
                    resultframe = results[0]
                    resultframe2 = results[1]
                    cv2.imshow("Person1", resultframe)
                    cv2.imshow("Person2", resultframe2)
                    pastframes1.clear()
                    pastframes2.clear()
                    cv2.waitKey(100)

                else:
                    client.send_message("/filter", 0)
                    resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                    resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("Person1", resultframe)
                    cv2.imshow("Person2", resultframe2)
                    cv2.waitKey(100)
            

            else: # if match result not true but there are two faces, just draw eyes and update start time
                client.send_message("/filter", 0)
                start_time = time.time() 
                resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
                resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

                if len(video_process.rightpupils) ==2 and len(video_process.rightpupils2) ==2:

                    firstlefteye_region = np.array([(video_process.landmarks.part(36).x, video_process.landmarks.part(36).y), (video_process.landmarks.part(37).x, video_process.landmarks.part(37).y), (video_process.landmarks.part(38).x, video_process.landmarks.part(38).y), (video_process.landmarks.part(39).x, video_process.landmarks.part(39).y), (video_process.landmarks.part(40).x, video_process.landmarks.part(40).y), (video_process.landmarks.part(41).x, video_process.landmarks.part(41).y)], np.int32)
                    firstrighteye_region = np.array([(video_process.landmarks.part(42).x, video_process.landmarks.part(42).y), (video_process.landmarks.part(43).x, video_process.landmarks.part(43).y), (video_process.landmarks.part(44).x, video_process.landmarks.part(44).y), (video_process.landmarks.part(45).x, video_process.landmarks.part(45).y), (video_process.landmarks.part(46).x, video_process.landmarks.part(46).y), (video_process.landmarks.part(47).x, video_process.landmarks.part(47).y)], np.int32)
                    secondlefteye_region = np.array([(video_process.landmarks2.part(36).x, video_process.landmarks2.part(36).y), (video_process.landmarks2.part(37).x, video_process.landmarks2.part(37).y), (video_process.landmarks2.part(38).x, video_process.landmarks2.part(38).y), (video_process.landmarks2.part(39).x, video_process.landmarks2.part(39).y), (video_process.landmarks2.part(40).x, video_process.landmarks2.part(40).y), (video_process.landmarks2.part(41).x, video_process.landmarks2.part(41).y)], np.int32)
                    secondrighteye_region = np.array([(video_process.landmarks2.part(42).x, video_process.landmarks2.part(42).y), (video_process.landmarks2.part(43).x, video_process.landmarks2.part(43).y), (video_process.landmarks2.part(44).x, video_process.landmarks2.part(44).y), (video_process.landmarks2.part(45).x, video_process.landmarks2.part(45).y), (video_process.landmarks2.part(46).x, video_process.landmarks2.part(46).y), (video_process.landmarks2.part(47).x, video_process.landmarks2.part(47).y)], np.int32)
                    cv2.polylines(resultframe2,[firstlefteye_region], True, (255), 2 )
                    cv2.polylines(resultframe2,[firstrighteye_region], True, (255), 2 )
                    cv2.polylines(resultframe,[secondlefteye_region], True, (255), 2 )
                    cv2.polylines(resultframe,[secondrighteye_region], True, (255), 2 )

                else:
                    drawingeyes = False
                #resultframe = cv2.resize(resultframe, 200, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Person1", resultframe)
                cv2.imshow("Person2", resultframe2)
                cv2.waitKey(150) #update frame and draw eyes every 150ms if not match

        else: # if no two faces = IDLE

            client.send_message("/filter", 0) #inform max that connection is broken
            drawingeyes = False
            #following two lines cause the flip between screens, now not active
            #resultframe = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 
            #resultframe2 = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
            #these show the same frame in one screen all the time, whether people are present or not
            resultframe = cv2.cvtColor(vidframe, cv2.COLOR_BGR2GRAY) 
            resultframe2 = cv2.cvtColor(vidframe2, cv2.COLOR_BGR2GRAY) 

            cv2.imshow("Person1", resultframe)
            cv2.imshow("Person2", resultframe2)
            cv2.waitKey(500) #update frame every 3 seconds if no-one is around. still checks faces all the time. 
            #THIS WAITKEY NUMBER MIGHT NEED TO BE SMALLER so that a glicth in face detection doesnt freeze screen for 3 secs


        key = cv2.waitKey(1)
        if key == 27: #esc
            break # escape loop
            cv2.destroyAllWindows()

    # Release handle to the webcam
    video_capture.stop()
    video_capture2.stop()
    video_process.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


main()