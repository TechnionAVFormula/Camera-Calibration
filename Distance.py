import numpy as np
import cv2
from numpy.linalg import inv

# Some how get R,t,K

def World_XY_from_uv_and_Z(imgpoints, K, R, t, Z):
    imgpoints_h = cv2.convertPointsToHomogeneous(imgpoints) # Turns the 2d cordinate (u,v) to a 3d (u,v,1)
    Rinv = inv(R)
    Kinv = inv(K)
    objpoints = np.zeros((len(imgpoints), 3))  # Possible to give type >>> dtype=np.float64

    obj_vector = Rinv @ t

    for i in range (len(imgpoints)):
        uv = np.squeeze(imgpoints_h[i])
        img_vector = Rinv @ Kinv @ uv

        s = (Z + obj_vector[2])/(img_vector[2])
        obj_pos = Rinv @ (s * Kinv @ uv - t)

        objpoints[i] = obj_pos

    return objpoints
