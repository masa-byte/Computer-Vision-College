import numpy as np
import cv2 as cv

bodyDetector = cv.CascadeClassifier("haarcascade_fullbody.xml")

cap = cv.VideoCapture('TownCentre_720p30.mkv')
while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    bodies = bodyDetector.detectMultiScale(gray)
    for (x, y, w, h) in bodies:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('Output', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
