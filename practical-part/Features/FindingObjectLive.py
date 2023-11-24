import numpy as np
import cv2 as cv

SIFT_LIKE = True
MIN_MATCH_COUNT = 15

imgT = cv.imread('tvrdjava.png')

# Initiate detector
if SIFT_LIKE:
    detector = cv.SIFT_create()
    # !!! pip3 install --upgrade opencv-contrib-python==3.4.2.17
    #detector = cv.xfeatures2d.SURF_create()
else:
    #detector = cv.ORB_create()
    #detector = cv.AKAZE_create()
    detector = cv.BRISK_create()

# Initiate matcher
if SIFT_LIKE:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
else:
    # create BFMatcher object
    matcher = cv.BFMatcher()#cv.NORM_HAMMING)

# Find the keypoints and descriptors
kpT, desT = detector.detectAndCompute(imgT, None)

# Template image properties
h, w = imgT.shape[:2]    
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

cap = cv.VideoCapture(0)
while True:
    ret, img = cap.read()

    kp, des = detector.detectAndCompute(img, None)

    # Matching descriptors
    matches = matcher.knnMatch(desT, des, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    matchesMask = None
    matchColor = (0, 0, 255)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpT[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is not None:
            matchesMask = mask.ravel().tolist()            
            dst = cv.perspectiveTransform(pts, M)
            img = cv.polylines(img, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
            matchColor = (0, 255, 0)
    
    imgOut = cv.drawMatches(imgT, kpT, img, kp, good, None, matchColor=matchColor, singlePointColor=None,
                            matchesMask=matchesMask,
                            flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Output", imgOut)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destoryAllWindows()
