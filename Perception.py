import numpy as np
import cv2
import pyzed.sl as sl
from ReadData import read_alignment, read_calibration
from Distance import  World_XY_from_uv_and_Z
from utils import getBoxes

def Init():
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

    return zed, image_left, runtime_parameters

def Record(zed, image_left, runtime_parameters):
    
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

        # Close the camera
        #zed.close()

        return img

zed, image_left, runtime_parameters = Init()

while True:
    
    img = Record(zed, image_left, runtime_parameters)

    #cv2.imshow("Image", img)
    # cv2.imwrite('Cones_img.png', img)
    #cv2.waitKey(100)

    imgpoints = getBoxes(img)
    u = [pixel[0] for pixel in imgpoints]
    v = [pixel[1] for pixel in imgpoints]
    imgpoints = [u, v]
    imgpixels = np.array(imgpoints, dtype=np.int)
    imgpoints = np.array(imgpoints, dtype=np.float64)

    # Load the camera parameters: 
    # K - Camera matrix, d - Distortion coefficients vector,
    # R - Rotation matrix, t - Translation vector.

    K, d = read_calibration()
    R, t = read_alignment()

    N = imgpoints.shape[1]
    # Undistort use (1,N,2) shape so need to reshape the vector: 
    imgpoints1 = imgpoints.reshape(1,N,2)
    points_undist = cv2.undistortPoints(imgpoints1, cameraMatrix=K, distCoeffs=d, dst=None, R=None, P=np.eye(3))
    points_undist = points_undist.reshape(N,2)  # there is an extra level of array which we no longer need

    # Note that we passed in an identity matrix as the new camera matrix ("P"). We can pass
    # in any valid intrinsics matrix. The undistort function remaps the points to the new projection.
    # We have Z=0 since we chose points on the floor.
    positions = World_XY_from_uv_and_Z(points_undist, K=np.eye(3), R=R, t=t.reshape(3,1), Z=0.0)
    for i in range(len(positions)):
        cv2.putText(img, "x:{}, y:{}".format(positions[i][0], positions[i][1]),tuple(imgpixels[:,i]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0,255,0))

    cv2.imshow('a',img)
    cv2.waitKey(1500)




