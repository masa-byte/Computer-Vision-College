import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

MIN_MATCH_COUNT = 15


def load_images():
    folder = os.path.dirname(os.path.abspath(__file__))
    images = []
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    return images


def show_plot(images, titles, cmap=True, method=1):
    plt.figure(figsize=(10, 10))
    if method == 1:
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i], cmap="gray" if cmap else None)
            plt.title(titles[i] if len(titles) > i else None)
    else:
        for i in range(len(images)):
            plt.subplot(len(images), len(images) // len(images), i + 1)
            plt.imshow(images[i], cmap="gray" if cmap else None)
            plt.title(titles[i] if len(titles) > i else None)
    plt.show()


def warp_images(img1, img2, M):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, M)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Mt = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, Mt @ M, (xmax - xmin, ymax - ymin))
    warped_img2[t[1] : h1 + t[1], t[0] : w1 + t[0]] = img1

    return warped_img2


def create_panorama(warped_images, homography_matrices):
    panorama_width = max(warped.shape[1] for warped in warped_images)
    panorama_height = max(warped.shape[0] for warped in warped_images)

    # Create a blank canvas for the panorama
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # Iterate over the warped images and place them onto the canvas
    for warped, M in zip(warped_images, homography_matrices):
        # Warp the current image onto the panorama
        warped_image = cv2.warpPerspective(warped, M, (panorama_width, panorama_height))

        # Create a mask for the current image
        mask = np.zeros_like(warped_image, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(M.T[0:2].T)], (255, 255, 255))

        # Blend the current image onto the panorama
        panorama = cv2.addWeighted(panorama, 1, warped_image, 1, 0)

    return panorama


def flannMatcherSift(img1, img2):
    detected_keypoints = []
    warped_image = None

    imgKeyPoints = img1.copy()
    imgKeyPoints2 = img2.copy()

    detector = cv2.SIFT_create()

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    for point in kp1:
        imgKeyPoints = cv2.circle(
            imgKeyPoints,
            (int(point.pt[0]), int(point.pt[1])),
            radius=2,
            color=(255, 0, 0),
            thickness=-1,
        )
    detected_keypoints.append(imgKeyPoints)

    for point in kp2:
        imgKeyPoints2 = cv2.circle(
            imgKeyPoints2,
            (int(point.pt[0]), int(point.pt[1])),
            radius=2,
            color=(255, 0, 0),
            thickness=-1,
        )
    detected_keypoints.append(imgKeyPoints2)

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
        matchesMask = mask.ravel().tolist()

        warped_image = warp_images(img2, img1, M)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2,
    )

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    show_plot(detected_keypoints, ["Detected keypoints"] * 2, cmap=True, method=1)
    show_plot([img3], ["Matched image"], cmap=False, method=2)
    show_plot([warped_image], ["Warped image"], cmap=False, method=1)
    return warped_image


def compute(images):
    length = len(images)
    warped_image = flannMatcherSift(images[0], images[1])

    for i in range(2, length):    
        panoramic_image = flannMatcherSift(warped_image, images[i])
        warped_image = panoramic_image
    return panoramic_image


if __name__ == "__main__":
    images = load_images()
    panoramic_image = compute(images)
