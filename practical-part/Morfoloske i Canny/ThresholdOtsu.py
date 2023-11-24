import numpy as np
import cv2 as cv


cap = cv.VideoCapture(0)
while True:
    _, img = cap.read()

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 10)

    cv.imshow("Output", img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
