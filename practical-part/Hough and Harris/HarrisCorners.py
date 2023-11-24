import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while True:    
    ret, img = cap.read()    
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = gray.astype(np.float32)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dstScale = 255 / dst.max()
    dstOut = dst * dstScale
    dstOut = np.clip(dstOut, 0, 255)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.        
    img[dst > 0.1 * dst.max()] = [0, 0, 255]
        
    cv.imshow('Cornerness', dstOut)
    cv.imshow('Output', img)  
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
