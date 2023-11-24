import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize


imgIn = cv.imread("s58Hl.jpg")
imgGray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
retval, img = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("Input", img)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(11, 11))

imgErode = cv.erode(img, kernel=kernel)
cv.imshow("Erode", imgErode)

imgDilate = cv.dilate(img, kernel=kernel)
cv.imshow("Dilate", imgDilate)

imgOpen = cv.erode(img, kernel=kernel)
imgOpen = cv.dilate(imgOpen, kernel=kernel)
cv.imshow("Open", imgOpen)

imgOpen2 = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel)
cv.imshow("Open2", imgOpen2)

imgClose = cv.dilate(img, kernel=kernel)
imgClose = cv.erode(imgClose, kernel=kernel)
cv.imshow("Close", imgClose)

imgClose2 = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel)
cv.imshow("Close2", imgClose2)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
imgEdge = cv.dilate(img, kernel=kernel)
imgEdge = imgEdge - img
cv.imshow("Edge", imgEdge)


def thinning(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv.erode(img, kernel)
        temp = cv.dilate(eroded, kernel)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True

    return skel


img = cv.imread('Fingerprint.jpg', 0)
_, imgTh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("Fingerprint", imgTh)

imgThinning = thinning(imgTh)
cv.imshow("Thinning", imgThinning)

imgSkeletonize = skeletonize(imgTh // 255).astype(np.uint8) * 255
cv.imshow("Skeletonize", imgSkeletonize)

cv.waitKey(0)
cv.destroyAllWindows()
