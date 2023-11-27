import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os

def load_images():
    folder = os.path.dirname(os.path.abspath(__file__))
    images = []
    for file in os.listdir(folder):
        if file.endswith(".png") and "box" in file:
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

    return images

if __name__ == '__main__':

    MIN_MATCH_COUNT = 10

    images = load_images()

    img1 = images[0]  # queryImage
    img2 = images[1] # trainImage

    # Initiate SIFT detector
    detector = cv2.SIFT_create()

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    for point in kp1:
        img1 = cv2.circle(img1, (int(point.pt[0]), int(point.pt[1])), radius=2, color=(0, 0, 0), thickness=-1)
    plt.imshow(img1, cmap='gray')
    plt.show()
    print(kp1[0].pt[0], kp1[0].pt[1])
    print(des1.shape)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(M)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3)
    plt.show()