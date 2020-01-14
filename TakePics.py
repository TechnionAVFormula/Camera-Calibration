import pyzed.sl as sl
import sys
import os
import cv2
#import time


### def main():

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

### Enable recording with the filename specified in argument
### path_output = sys.argv[0]
### path_output = os.getcwd() + path_output
### err = zed.enable_recording(path_output, sl.SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS)


# Create a Camera object
image_left = sl.Mat()
###image_right = sl.Mat()
runtime_parameters = sl.RuntimeParameters()
# Capture 50 frames and stop
for i in range(50):
    ### input()
    # Grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        ### Each new frame is added to the SVO file
        ### zed.record()
        zed.retrieve_image(image_left, sl.VIEW.VIEW_LEFT)
        ###zed.retrieve_image(image_right, sl.VIEW.VIEW_RIGHT)
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT)  # Get the timestamp at the time the image was captured
        print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image_left.get_width(), image_left.get_height(),
                timestamp))

        # To recover data from sl.Mat to use it with opencv, we use the get_data() method
        # It returns a numpy array that can be used as a matrix with opencv
        image_ocv = image_left.get_data()
        ###depth_image_ocv = depth_image_zed.get_data()

        cv2.imshow("Image", image_ocv)
        cv2.imwrite('50images/{}.png'.format(i),image_ocv)
        cv2.waitKey(2500)
        ###cv2.imshow("Depth", depth_image_ocv)
        

# Disable recording
### zed.disable_recording()

# Close the camera
zed.close()

### if __name__ == "__main__":
###     main()