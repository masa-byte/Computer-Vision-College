import numpy as np
import cv2


def nothing(x):
    pass


wndOut = cv2.namedWindow("Output")
tbTh1 = cv2.createTrackbar("Th1", "Output", 100, 1500, nothing)
tbTh2 = cv2.createTrackbar("Th2", "Output", 200, 1500, nothing)

#imgIn = cv2.imread("Circles.jpg")
imgIn = cv2.imread("body_10-buildings_1.JPG")
#"s58Hl.jpg")#"body_10-buildings_1.JPG")#"170508_10_27_41_5DS29248.0.jpg")
imgGray = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)

while True:
    th1 = cv2.getTrackbarPos("Th1", "Output")
    th2 = cv2.getTrackbarPos("Th2", "Output")
    imgEdge = cv2.Canny(imgGray, th1, th2, apertureSize=3, L2gradient=True)

    cv2.imshow("Input", imgGray)
    cv2.imshow("Output", imgEdge)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
