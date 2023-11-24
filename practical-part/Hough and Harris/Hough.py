import numpy as np
import cv2
import matplotlib.pyplot as plt


def circles(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=img_gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=10, param2=20,
                               minRadius=10, maxRadius=20)
    for circle in circles[0]:
        img = cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0))
    plt.imshow(img)
    plt.show()


def lines(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(img_gray, 100, 200)
    plt.imshow(mask)
    plt.show()
    lines = cv2.HoughLinesP(image=mask, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    print(lines)
    for line in lines:
        img = cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color=(255, 0, 255))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # img = cv2.imread("s58Hl.jpg")
    # circles(img)
    img = cv2.imread("body_10-buildings_1.JPG")
    # lines(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    perc_90 = np.percentile(result, 99)
    _, mask = cv2.threshold(result, thresh=perc_90, maxval=np.max(result), type=cv2.THRESH_BINARY)
    plt.imshow(mask)
    plt.show()
