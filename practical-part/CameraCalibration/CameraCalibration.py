import numpy as np
import cv2 as cv
import glob

# Chessboard size (number of crosses and not fields!)
board_width = 31
board_height = 23

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...., (30,22,0)
objp = np.zeros((board_width * board_height, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('camera_calib*.jpg')
width, height = 0, 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (board_width, board_height),
                                            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # If found, add object points, image points (after refining them)
        objpoints.append(objp)

        termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (board_width, board_height), corners2, ret)
        cv.imshow(fname, img)
        cv.waitKey(1)

# Calibrate camera
#    cameraMatix - 3x3 intrinsic matrix
#    distCoeffs - distortion coefficients
#    rvects - list of rotation vectors (extracted from rotation matrix, see cv.Rodrigues2) for each image
#    tvects - list of translation vectors for each image
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize=(width, height),
                                                                 cameraMatrix=None, distCoeffs=None)
print('cameraMatrix', cameraMatrix)
print('distCoeffs', distCoeffs)
print('rvecs', rvecs)
print('tvecs', tvecs)

# Calculate mean error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("Total error: {}".format(mean_error / len(objpoints)))

# Reproject images
for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newCameraMatrix, validPixROI = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
    print('validPixROI', validPixROI)

    if True:
        # undistort
        dst = cv.undistort(img, cameraMatrix, distCoeffs, newCameraMatrix=newCameraMatrix)
    else:
        # undistort
        mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), cv.CV_32FC1)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = validPixROI
    dst = dst[y:y+h, x:x+w]

    # save and show the image
    cv.imwrite('Calibrated/' + fname, dst)
    cv.imshow('Calibrated/' + fname, dst)
    cv.waitKey(1)
    
cv.waitKey(0)
cv.destroyAllWindows()
