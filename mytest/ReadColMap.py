#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from cv2 import DISOpticalFlow
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import sys

# In[ ]:


class Reconstruction:
    def __init__(self):
        self.cameras = {}
        self.views = {}
        self.points3d = {}
        self.min_view_id = -1
        self.max_view_id = -1
        self.image_folder = ""
    
    def ViewIds(self):
        return list(self.views.keys())
    
    def GetNeighboringKeyframes(self, view_id):
        previous_keyframe = -1
        next_keyframe = -1
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                previous_keyframe = idx
                break
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                next_keyframe = idx
                break
        if previous_keyframe < 0 or next_keyframe < 0:
            return np.array([])
        return [previous_keyframe, next_keyframe]
    
    def GetReferenceFrames(self, view_id):
        kf = self.GetNeighboringKeyframes(view_id)
        if (len(kf) < 2):
            return []
        dist = np.linalg.norm(self.views[kf[1]].Position() -                              self.views[kf[0]].Position()) / 2
        pos = self.views[view_id].Position()
        ref = []
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos -                              self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos -                              self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        return ref

    def GetImage(self, view_id):
        return self.views[view_id].GetImage(self.image_folder)
    
    def GetSparseDepthMap(self, frame_id):
        camera = self.cameras[self.views[frame_id].camera_id]
        view = self.views[frame_id]
        view_pos = view.Position()
        depth_map = np.zeros((camera.height, camera.width), dtype=np.float32)
        for point_id, coord in view.points2d.items():
            if (point_id>=0) and (coord[1]<=1920) and (coord[0]<=1080):
                pos3d = self.points3d[point_id].position3d
                depth = np.linalg.norm(pos3d - view_pos)
                depth_map[int(coord[1]), int(coord[0])] = depth
        return depth_map
    
    def Print(self):
        print("Found " + str(len(self.views)) + " cameras.")
        for id in self.cameras:
            self.cameras[id].Print()
        print("Found " + str(len(self.views)) + " frames.")
        for id in self.views:
            self.views[id].Print()

class Point:
    def __init__(self):
        self.id = -1
        self.position3d = np.zeros(3, float)
    
            
class Camera:

    def __init__(self):
        self.id = -1
        self.width = 0
        self.height = 0
        self.focal = np.zeros(2,float)
        self.principal = np.zeros(2,float)
        self.model = ""
    
    def Print(self):
        print("Camera " + str(self.id))
        print("-Image size: (" + str(self.width) +             ", " + str(self.height) + ")")
        print("-Focal: " + str(self.focal))
        print("-Model: " + self.model)
        print("")

class View:    
    def __init__(self):
        self.id = -1
        self.orientation = Quaternion()
        self.translation = np.zeros(3, float)
        self.points2d = {}
        self.camera_id = -1
        self.name = ""
    
    def IsKeyframe(self):
        return len(self.points2d) > 0
    
    def Rotation(self):
        return self.orientation.rotation_matrix
    
    def Position(self):
        return self.orientation.rotate(self.translation)
    
    def GetImage(self, image_folder):
        mat = cv2.imread(image_folder + "/" + self.name)
        # Check that we loaded correctly.
        assert mat is not None,             "Image " + self.name + " was not found in "             + image_folder
        return mat
    
    def Print(self):
        print("Frame " + str(self.id) + ": " + self.name)
        print("Rotation: \n" +             str(self.Rotation()))
        print("Position: \n" +             str(self.Position()))
        print("Number of points in this view: ",
              len(self.points2d))
        print("")
        
def ReadColmapCamera(filename):
    file = open(filename, "r")
    line = file.readline()
    cameras = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            cameras[id_value] = Camera()
            cameras[id_value].id = id_value
            cameras[id_value].model = tokens[1]
            # Currently we're assuming that the camera model
            # is in the SIMPLE_RADIAL format
            #assert(cameras[id_value].model == "PINHOLE")
            cameras[id_value].width = int(tokens[2])
            cameras[id_value].height = int(tokens[3])
            cameras[id_value].focal[0] = float(tokens[4])
            cameras[id_value].focal[1] = float(tokens[5])
            cameras[id_value].principal[0] = float(tokens[6])
            cameras[id_value].principal[1] = float(tokens[7])
        line = file.readline()
    return cameras;

def ReadColmapImages(filename):
    file = open(filename, "r")
    line = file.readline()
    views = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            views[id_value] = View()
            views[id_value].id = id_value
            views[id_value].orientation = Quaternion(float(tokens[1]),                                                      float(tokens[2]),                                                      float(tokens[3]),                                                      float(tokens[4]))
            views[id_value].translation[0] = float(tokens[5])
            views[id_value].translation[1] = float(tokens[6])
            views[id_value].translation[2] = float(tokens[7])
            views[id_value].camera_id = int(tokens[8])
            views[id_value].name = tokens[9]
            line = file.readline()
            tokens = line.split()
            views[id_value].points2d = {}
            for idx in range(0, len(tokens) // 3):
                point_id = int(tokens[idx * 3 + 2])
                coord = np.array([float(tokens[idx * 3 + 0]),                          float(tokens[idx * 3 + 1])])
                views[id_value].points2d[point_id] = coord
            
            # Read the observations...
        line = file.readline()
    return views
           
def ReadColmapPoints(filename):
    file = open(filename, "r")
    line = file.readline()
    points = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            points[id_value] = Point()
            points[id_value].id = id_value
            points[id_value].position3d = np.array([float(tokens[1]),                                         float(tokens[2]),                                         float(tokens[3])])
            
        line = file.readline()
    return points
        
            
    
def ReadColmap(poses_folder, images_folder):
    # Read the cameras (intrinsics)
    recon = Reconstruction()
    recon.image_folder = images_folder
    recon.cameras = ReadColmapCamera(poses_folder + "FB_config/cameras.txt")
    recon.views = ReadColmapImages(poses_folder + "FB_config/images.txt")
    recon.points3d = ReadColmapPoints(poses_folder + "FB_config/points3D.txt")
    recon.min_view_id = min(list(recon.views.keys()))
    recon.max_view_id = max(list(recon.views.keys()))
    print("Number of points: " + str(len(recon.points3d)))
    print("Number of frames: " + str(len(recon.views)))
    #assert len(recon.views) == (recon.max_view_id - recon.min_view_id) + 1, "Min\max: " + str(recon.max_view_id) + " " + str(recon.min_view_id)
    return recon


# In[ ]:

'''
recon = ReadColmap(input_colmap, input_frames)
# I use 000029.png to demonstrate
# The conresponding idx in Views is 28
# 26.png -> idx 24
# 29.png -> idx 28
testFrameidx = 28
testfile = "000029"
testCase = recon.views[testFrameidx]
testCase.Print()
testImg = cv2.imread(input_frames+testfile+".png")
h, w, ch = testImg.shape
sparseDepthImg = np.zeros((h, w, ch))
print(sparseDepthImg.shape)
for key, value in testCase.points2d.items():
    cv2.circle(sparseDepthImg, (int(value[0]), int(value[1])), 15, (0,0,255), -1)
depth = recon.GetSparseDepthMap(testFrameidx)
idxNoneZeroRow, idxNoneZeroCol = np.nonzero(depth)



# In[ ]:


cv2.namedWindow("Test_Img", cv2.WINDOW_NORMAL)
cv2.imshow("Test_Img",sparseDepthImg)
cv2.waitKey(10000)
cv2.destroyWindow("Test_Img")

cv2.imwrite(testfile+"_sparse.png",sparseDepthImg)
'''


# In[ ]:




