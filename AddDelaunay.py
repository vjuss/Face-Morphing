from threading import Thread
import cv2
from VideoGet import VideoGet
from CheckFaceLoc import CheckFaceLoc
import numpy as np
import dlib

class AddDelaunay:
    @staticmethod
    def extract_index_nparray(nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    @staticmethod
    def translate(value, leftMin, leftMax, rightMin, rightMax): #https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)



    def __init__(self, srcframe, destframe, srcfaces, destfaces, srclandmarks, destlandmarks, elapsedtime):
        self.src_frame = srcframe
        self.dest_frame = destframe
        self.src_faces = srcfaces
        self.dest_faces = destfaces
        self.src_landmarks = srclandmarks 
        self.dest_landmarks = destlandmarks
        self.elapsed_time = elapsedtime
        self.resultframe = []
        self.resultframe2 = []
        self.stopped = False

    def process(self):
        while not self.stopped:
           
            sourceframe = self.src_frame
            destinationframe = self.dest_frame
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

            sourcefaces = self.src_faces
            destinationfaces = self.dest_faces

            for sourceface in sourcefaces:
                sourcelandmarks = self.src_landmarks
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
                    index_pt1 = AddDelaunay.extract_index_nparray(index_pt1)

                    index_pt2 = np.where((source_triangle_points == pt2).all(axis=1))
                    index_pt2 = AddDelaunay.extract_index_nparray(index_pt2)

                    index_pt3 = np.where((source_triangle_points == pt3).all(axis=1))
                    index_pt3 = AddDelaunay.extract_index_nparray(index_pt3)
                    
                    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                        source_triangle = [index_pt1, index_pt2, index_pt3]
                        indexes_triangles.append(source_triangle)

            # Face 2
            for destinationface in destinationfaces:
                destinationlandmarks = self.dest_landmarks
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

             ##elapsed_time = self.elapsed_time
             ##opacity = AddDelaunay.translate(elapsed_time, 0, 10, 0, 255)
             ##print("map wth function", opacity)

            final_destination_canvas = np.zeros_like(destinationgray)
            final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destinationconvexhull, 255) 
            final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)
            destination_face_masked = cv2.bitwise_and(destinationframe, destinationframe, mask=final_destination_canvas)
            result = cv2.add(destination_face_masked, destination_image_canvas)
            

            # Put reconstructed face on the source image

            final_source_canvas = np.zeros_like(sourcegray)
            final_source_face_mask = cv2.fillConvexPoly(final_source_canvas, sourceconvexhull, 255) #EDITED
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
            self.resultframe = resultframe 
            self.resultframe2 = resultframe2

    def stop(self):
         self.stopped = True


 
    

