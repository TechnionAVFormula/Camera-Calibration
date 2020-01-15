import pyzed.sl as sl
import cv2
import numpy as np
from ReadData import read_calibration

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080  # Use HD1080 video mode
init_params.camera_fps = 30  # Set fps at 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

image_left = sl.Mat()
runtime_parameters = sl.RuntimeParameters()

# Capture 1 frame for Alignment
 
# Grab an image, a RuntimeParameters object must be given to grab()
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # A new image is available if grab() returns SUCCESS
    # Each new frame is added to the SVO file
    
    zed.retrieve_image(image_left, sl.VIEW.VIEW_LEFT)
    # Get the timestamp at the time the image was captured
    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT)  
    print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image_left.get_width(), image_left.get_height(),
            timestamp))

    # To recover data from sl.Mat to use it with opencv, we use the get_data() method
    # It returns a numpy array that can be used as a matrix with opencv
    img = image_left.get_data()

    cv2.imshow("Image", img)
    cv2.waitKey(2500)

    # Close the camera
    zed.close()

# termination criteria - of the form: (type, max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (3,0,0), (6,0,0) ....,(21,12,0) - cordinate of the chessboard
#need to specify the correct chessboard grid size - here it is 8x5 and square size is 3cm.
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = (np.mgrid[0:8,0:5].T.reshape(-1,2))*3

# Arrays to store object points and image points from the image.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

###img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners for each image. return the corners points in image
# and retval will be true if the pattern is obtained
# Need to specify the correct chessboard pattern size of inner corners - here it is 8x5
# Note: The function requires white space (like a square-thick border, the wider the better) around
# the board to make the detection more robust in various environments. 
ret, corners = cv2.findChessboardCorners(gray, (8,5),None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp) 

    # increase the accuracy of the corners
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
    
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (8,5), corners2,ret)
    cv2.imshow('img',img)
    cv2.waitKey(2500)

# Returns the camera matrix, distortion coefficients, rotation and translation vectors.
# rvec - Axis with angle magnitude (radians) [x, y, z]

R, d = read_calibration()

retval, rvec, tvec = cv2.solvePnP(objectPoints=objpoints[0], imagePoints=imgpoints[0], cameraMatrix=R, distCoeffs=d)
# ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# Transform the rotation vector to Rmat - Rotation Matrix (radians)
Rmat,_ = cv2.Rodrigues(rvec)

# Saving the paramaters
np.savez('Alignment.npz', **{'RotationMtx': Rmat, 'TranslationVec': tvec})