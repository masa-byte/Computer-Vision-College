import cv2 as cv


#####################################################################################


def bfMatcherSift():
    img1 = cv.imread('box.png', 0)           # queryImage
    img2 = cv.imread('box_in_scene.png', 0)  # trainImage
    
    # Initiate SIFT detector
    detector = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Output", img3)
    cv.waitKey(0)


#####################################################################################


def flannMatcherSift():
    img1 = cv.imread('box.png', 0)           # queryImage
    img2 = cv.imread('box_in_scene.png', 0)  # trainImage
    
    # Initiate SIFT detector
    detector = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
            
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    cv.imshow("Output", img3)
    cv.waitKey(0)


#####################################################################################


def flannMatcherSurf():
    img1 = cv.imread('box.png', 0)  # queryImage
    img2 = cv.imread('box_in_scene.png', 0)  # trainImage

    # Initiate SURF detector
    detector = cv.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Output", img3)
    cv.waitKey(0)


#####################################################################################


def bfMatcherOrb():    
    img1 = cv.imread('box.png', 0)           # queryImage
    img2 = cv.imread('box_in_scene.png', 0)  # trainImage

    # Initiate ORB detector
    detector = cv.ORB_create()
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    cv.imshow("Output", img3)
    cv.waitKey(0)


#####################################################################################


def bfMatcherAkaze():
    img1 = cv.imread('box.png', 0)  # queryImage
    img2 = cv.imread('box_in_scene.png', 0)  # trainImage

    # Initiate AKAZE detector
    detector = cv.AKAZE_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Output", img3)
    cv.waitKey(0)


#####################################################################################


def bfMatcherBrisk():
    img1 = cv.imread('box.png', 0)  # queryImage
    img2 = cv.imread('box_in_scene.png', 0)  # trainImage

    # Initiate BRISK detector
    detector = cv.BRISK_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Output", img3)
    cv.waitKey(0)


bfMatcherSift()
#flannMatcherSift()
#flannMatcherSurf()
#bfMatcherOrb()
#bfMatcherAkaze()
#bfMatcherBrisk()
