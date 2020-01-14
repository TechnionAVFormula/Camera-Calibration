import numpy as np
import cv2
import pyzed.sl as sl
from ReadData import read_alignmnet, read_calibration
from Distance import  World_XY_from_uv_and_Z


img = Take_Pic()

cv2.imshow("Image", img)
cv2.imwrite('Cones_img.png')
cv2.waitKey(2500)

### imgpoints = Cone detection() ###

# Load the camera parameters: 
# K - Camera matrix, d - Distortion coefficients vector,
# R - Rotation matrix, t - Translation vector.
K, d = read_calibration()
R, t = read_alignmnet()

### Undistort use (1,N,2) shape so maybe need to reshape the vector: imgpoints.reshape(1,N,2)
### To get back the normal form do: points_undist[0] or points_undist.reshape(N,2)
points_undist = cv2.undistortPoints(imgpoints, K=K, D=d, R=None, P=np.eye(3))
points_undist = points_undist[0]  # there is an extra level of array which we no longer need

# Note that we passed in an identity matrix as the new camera matrix ("P"). We can pass
# in any valid intrinsics matrix. The undistort function remaps the points to the new projection.
# We have Z=0 since we chose points on the floor.
positions = World_XY_from_uv_and_Z(points_undist, K=np.eye(3), R=R, t=t, Z=0.0)




def Take_Pic():
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
        # Close the camera
        zed.close()

        return img
