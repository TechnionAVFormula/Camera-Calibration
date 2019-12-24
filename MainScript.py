import numpy as np
import cv2
import glob

# termination criteria - of the form: (type, max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,4,0) - cordinate of the chessboard
#need to specify the correct chessboard grid size - here it is 8x5
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Return a possibly-empty list of path names that match pathname(*.jpg)
##### need to get images
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners for each image. return the corners points in image
    # and retval will be true if the pattern is obtained
    # need to specify the correct chessboard pattern size of inner corners?!? - here it is 8x5
    # note The function requires white space (like a square-thick border, the wider the better) around
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
        cv2.waitKey(500)

   
# returns the camera matrix, distortion coefficients, rotation and translation vectors.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


img = cv2.imread('left12.jpg')
h,  w = img.shape[:2]
# refine the camera matrix - If the scaling parameter alpha = 0,
# returns undistorted image with minimum unwanted pixels (may remove pixels at image corners).
# if alpha = 1, all pixels are retained with some extra black images.
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)


cv2.destroyAllWindows()

