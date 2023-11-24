import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# Ucitava i prikazuje originalnu sliku
img = cv.imread('s58Hl.jpg')
cv.imshow("Input", img)

# Median blur
imgMedian = cv.medianBlur(img, 11)
cv.imshow("Median blur", imgMedian)

# Prevodi u HSV i ekstraktuje Saturation kanal
imgHSV = cv.cvtColor(imgMedian, cv.COLOR_BGR2HSV)
imgSat = imgHSV[:, :, 1]
# Privremeno prikazuje sliku pomocu pyplot-a da bi se ocitao opseg hue vrednosti za pozadinu
if True:
    plt.imshow(imgSat)
    plt.show()

# Konvertuje sliku u binarnu koristeci opseg vrednosti
imgTh = cv.inRange(imgSat, 55, 255)
cv.imshow("Threshold", imgTh)

# Primenjuje morfolosku operaciju otvaranja da odvoji blobove koji se dodiruju
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
imgOpen = cv.morphologyEx(src=imgTh, op=cv.MORPH_OPEN, kernel=kernel)
cv.imshow("Open", imgOpen)

# Nalazi povezane komponente
imgOut = img.copy()
cntCC, imgCC = cv.connectedComponents(imgOpen, connectivity=4)

# Nalazi i crta bounding box za svaku od identifikovanih komponenti, nalazi najvecu komponentu
maxCnt = 0
maxBBox = None
for cc in range(1, cntCC):
    imgCurr = np.where(imgCC == cc, 255, 0).astype(np.uint8)
    x, y, w, h = cv.boundingRect(imgCurr)
    cnt = imgCurr.sum() / 255
    if cnt > maxCnt:
        maxCnt = cnt
        maxBBox = x, y, w, h
    cv.rectangle(imgOut, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)

# Ispisuje broj identifikovanih komponenti i prikazuje izlaz
cv.putText(imgOut, text='CNT: '+str(cntCC), org=(5, 17), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
           color=(0, 0, 255), thickness=2)
cv.imshow("Output", imgOut)

cv.waitKey(0)
cv.destroyAllWindows()
