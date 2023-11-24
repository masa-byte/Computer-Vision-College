import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while True:    
    ret, img = cap.read()    

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, 500, 0.1, 5)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 3, (0, 255, 255), -1)

    cv.imshow('Output', img)  

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
