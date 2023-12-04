"""
https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
"""
import numpy as np
import cv2 as cv
from imutils.object_detection import non_max_suppression

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture('TownCentre_720p30.mkv')
while True:
    ret, img = cap.read()
    img2 = img.copy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for rect, weight in zip(rects, weights):
        x, y, w, h = rect
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(img, str(weight), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv.LINE_AA)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv.rectangle(img2, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    cv.imshow("Before NMS", img)
    cv.imshow("After NMS", img2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
