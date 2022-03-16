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

    def __init__(self, frame, frame2, detector, predictor):
        self.src_frame = frame
        self.src_frame2 = frame2
        self.detector = detector
        self.predictor = predictor
        #self.seamlessclone = () CHEKC FORMAT 
        #self.seamlessclone2 = ()
        self.stopped = False

    def start(self):    
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:
           
            _frame = self.src_frame
            _frame2 = self.src_frame2

            _gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)  #for example this is repetition
            _gray2 = cv2.cvtColor(_frame2, cv2.COLOR_BGR2GRAY)

              #Find faces
            _detector = self.detector
            _predictor = self.predictor

        
            _faces= _detector(_gray)
            _faces2= _detector(_gray2)

            _mask = np.zeros_like(_gray)
            _height, _width, _channels = _frame2.shape
            _img2_new_face = np.zeros((_height, _width, _channels), np.uint8)
            _indexes_triangles = []

            _mask2 = np.zeros_like(_gray2)
            _height2, _width2, _channels2 = _frame.shape
            _img1_new_face = np.zeros((_height2, _width2, _channels2), np.uint8)
            _indexes_triangles2 = []

            #landmarks of first face

            for _face in _faces:
                _landmarks = _predictor(_gray, _face)
                _landmarks_points = []
                for n in range(0, 68):
                    _x = _landmarks.part(n).x
                    _y = _landmarks.part(n).y
                    _landmarks_points.append((_x, _y))

                _points = np.array(_landmarks_points, np.int32)

                _convexhull = cv2.convexHull(_points)
                cv2.fillConvexPoly(_mask, _convexhull, 255)    
                
                _face_image_1 = cv2.bitwise_and(_frame, _frame, mask=_mask)
        

                # Delaunay triangulation

                _rect = cv2.boundingRect(_convexhull)
                _subdiv = cv2.Subdiv2D(_rect)
                _subdiv.insert(_landmarks_points)
                _triangles = _subdiv.getTriangleList()
                _triangles = np.array(_triangles, dtype=np.int32)
                
                #indexes_triangles = []

                for _t in _triangles:
                    _pt1 = (_t[0], _t[1])
                    _pt2 = (_t[2], _t[3])
                    _pt3 = (_t[4], _t[5])

                    _index_pt1 = np.where((_points == _pt1).all(axis=1))
                    _index_pt1 = AddDelaunay.extract_index_nparray(_index_pt1)

                    _index_pt2 = np.where((_points == _pt2).all(axis=1))
                    _index_pt2 = AddDelaunay.extract_index_nparray(_index_pt2)

                    _index_pt3 = np.where((_points == _pt3).all(axis=1))
                    _index_pt3 = AddDelaunay.extract_index_nparray(_index_pt3)
                    
                    if _index_pt1 is not None and _index_pt2 is not None and _index_pt3 is not None:
                        _triangle = [_index_pt1, _index_pt2, _index_pt3]
                        _indexes_triangles.append(_triangle)

            # Face 2
            for _face in _faces2:
                _landmarks = _predictor(_gray2, _face)
                _landmarks_points2 = []
                for n in range(0, 68):
                    _x = _landmarks.part(n).x
                    _y = _landmarks.part(n).y
                    _landmarks_points2.append((_x, _y))


                _points2 = np.array(_landmarks_points2, np.int32)
                _convexhull2 = cv2.convexHull(_points2)
                cv2.fillConvexPoly(_mask2, _convexhull2, 255)  
                _face_image_2 = cv2.bitwise_and(_frame2, _frame2, mask=_mask2)


            # Creating empty mask
            _lines_space_mask = np.zeros_like(_gray)
            _lines_space_new_face = np.zeros_like(_frame2)

            # Triangulation of both faces
            for _triangle_index in _indexes_triangles:
                # Triangulation of the first face
                _tr1_pt1 = _landmarks_points[_triangle_index[0]]
                _tr1_pt2 = _landmarks_points[_triangle_index[1]]
                _tr1_pt3 = _landmarks_points[_triangle_index[2]]
                _triangle1 = np.array([_tr1_pt1, _tr1_pt2, _tr1_pt3], np.int32)


                _rect1 = cv2.boundingRect(_triangle1)
                (x, y, w, h) = _rect1  #check whether h or _h
                _cropped_triangle = _frame[y: y + h, x: x + w]
                _cropped_tr1_mask = np.zeros((h, w), np.uint8)


                _points = np.array([[_tr1_pt1[0] - x, _tr1_pt1[1] - y],
                                [_tr1_pt2[0] - x, _tr1_pt2[1] - y],
                                [_tr1_pt3[0] - x, _tr1_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(_cropped_tr1_mask, _points, 255)

                # Lines space
                cv2.line(_lines_space_mask, _tr1_pt1, _tr1_pt2, 255)
                cv2.line(_lines_space_mask, _tr1_pt2, _tr1_pt3, 255)
                cv2.line(_lines_space_mask, _tr1_pt1, _tr1_pt3, 255)
                _lines_space = cv2.bitwise_and(_frame, _frame, mask=_lines_space_mask)

                # Triangulation of second face
                _tr2_pt1 = _landmarks_points2[_triangle_index[0]]
                _tr2_pt2 = _landmarks_points2[_triangle_index[1]]
                _tr2_pt3 = _landmarks_points2[_triangle_index[2]]
                _triangle2 = np.array([_tr2_pt1, _tr2_pt2, _tr2_pt3], np.int32)


                _rect2 = cv2.boundingRect(_triangle2)
                (x, y, w, h) = _rect2 #check whether h or _h

                _cropped_tr2_mask = np.zeros((h, w), np.uint8)

                _points2 = np.array([[_tr2_pt1[0] - x, _tr2_pt1[1] - y],
                                    [_tr2_pt2[0] - x, _tr2_pt2[1] - y],
                                    [_tr2_pt3[0] - x, _tr2_pt3[1] - y]], np.int32)

                cv2.fillConvexPoly(_cropped_tr2_mask, _points2, 255)

                # Warp triangles
                _points = np.float32(_points)
                _points2 = np.float32(_points2)
                _M = cv2.getAffineTransform(_points, _points2)
                _warped_triangle = cv2.warpAffine(_cropped_triangle, _M, (w, h))
                _warped_triangle = cv2.bitwise_and(_warped_triangle, _warped_triangle, mask=_cropped_tr2_mask)

                # Reconstructing destination face
                _img2_new_face_rect_area = _img2_new_face[y: y + h, x: x + w] #check whether x or _x
                _img2_new_face_rect_area_gray = cv2.cvtColor(_img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, _mask_triangles_designed = cv2.threshold(_img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                _warped_triangle = cv2.bitwise_and(_warped_triangle, _warped_triangle, mask=_mask_triangles_designed)

                _img2_new_face_rect_area = cv2.add(_img2_new_face_rect_area, _warped_triangle)
                _img2_new_face[y: y + h, x: x + w] = _img2_new_face_rect_area

            #Face swapped (putting 1st face into 2nd face)
            _img2_face_mask = np.zeros_like(_gray2)
            _img2_head_mask = cv2.fillConvexPoly(_img2_face_mask, _convexhull2, 100) #255 is full opacity
            _img2_head_mask2 = cv2.fillConvexPoly(_img2_face_mask, _convexhull2, 220) #255 is full opacity
            _img2_face_mask = cv2.bitwise_not(_img2_head_mask)
            _img2_face_mask2 = cv2.bitwise_not(_img2_head_mask2)

            _img2_head_noface = cv2.bitwise_and(_frame2, _frame2, mask=_img2_face_mask)
            _img2_head_noface2 = cv2.bitwise_and(_frame2, _frame2, mask=_img2_face_mask2)
            _result = cv2.add(_img2_head_noface, _img2_new_face)
            _result2 = cv2.add(_img2_head_noface2, _img2_new_face)

            # Creating seamless clone of two faces
            (x, y, w, h) = cv2.boundingRect(_convexhull2)  #check whether h or _h
            _center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
            _seamlessclone = cv2.seamlessClone(_result, _frame2, _img2_head_mask, _center_face2, cv2.NORMAL_CLONE)
            _seamlessclone = cv2.cvtColor(_seamlessclone, cv2.COLOR_BGR2GRAY)

            _seamlessclone2 = cv2.seamlessClone(_result2, _frame2, _img2_head_mask2, _center_face2, cv2.NORMAL_CLONE)
            _seamlessclone2 = cv2.cvtColor(_seamlessclone2, cv2.COLOR_BGR2GRAY)

            #self.seamlessclone = _seamlessclone  #these will give results in the main function
            #self.seamlessclone2 = _seamlessclone2
            print("clone done") #this runs as a whole as clonedone is being printed
            print(_seamlessclone) #prints result like: [[210 210 210 ...  10  10  10]
                # [210 210 210 ...  10  10  10]
                # [210 210 209 ...  10  10  10]
                # ...
                # [186 189 192 ...   0   0   0]
                # [186 189 191 ...   0   0   0]
                # [186 188 189 ...   0   0   0]]


    def stop(self):
         self.stopped = True


 
    

