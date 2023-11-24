import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while True:    
    ret, img = cap.read()    

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    ##################################################
    # Feature detectors with descriptors
    
    detector = cv.SIFT_create()
    #detector = cv.xfeatures2d.SURF_create()
    #detector = cv.ORB_create()
    #detector = cv.BRISK_create()
    #detector = cv.KAZE_create()
    #detector = cv.AKAZE_create()

    kp, desc = detector.detectAndCompute(gray, mask=None)
    
    ##################################################
    # Feature Detectors without descriptors
                 
    #detector = cv.FastFeatureDetector_create()
    #detector = cv.AgastFeatureDetector_create()
    #detector = cv.MSER_create()
    #detector = cv.GFTTDetector_create()
    #detector = cv.SimpleBlobDetector_create()

    #kp = detector.detect(gray)
    
    cv.drawKeypoints(gray, kp, img, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow('Output', img)  

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
