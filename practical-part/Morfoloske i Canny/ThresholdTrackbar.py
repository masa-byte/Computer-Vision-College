import numpy as np
import cv2 as cv

imgIn = cv.imread("s58Hl.jpg")

#imgIn = cv.medianBlur(imgIn, 11)

imgHSV = cv.cvtColor(imgIn, cv.COLOR_BGR2HSV)

#cv.imshow("Hue", imgHSV[:, :, 0])
#cv.imshow("Saturation", imgHSV[:, :, 1])
#cv.imshow("Value", imgHSV[:, :, 2])
#cv.waitKey(0)

imgGray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
#imgGray = 255 - imgHSV[:, :, 1]


def refresh(x):
    _, imgOut = cv.threshold(imgGray, x, 255, cv.THRESH_BINARY_INV)
    cv.imshow("Output", imgOut)


wndOut = cv.namedWindow("Output")
tbTh = cv.createTrackbar("tbTh", "Output", 127, 255, refresh)

_, imgOut = cv.threshold(imgGray, 127, 255, cv.THRESH_BINARY)
imgOut = 255 - imgOut

cv.imshow("Input", imgIn)
cv.imshow("Gray", imgGray)
cv.imshow("Output", imgOut)
cv.waitKey(0)

cv.destroyAllWindows()
