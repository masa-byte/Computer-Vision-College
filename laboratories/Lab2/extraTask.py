import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images():
    folder = os.path.dirname(os.path.abspath(__file__))
    images = []
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
            images.append(img)

    return images


def show_plot(images, titles, cmap=True, method=1):
    plt.figure(figsize=(10, 10))
    if method == 1:
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i], cmap="gray" if cmap else None)
            plt.title(titles[i])
    else:
        for i in range(len(images)):
            plt.subplot(2, len(images) // 2, i + 1)
            plt.imshow(images[i], cmap="gray" if cmap else None)
            plt.title(titles[i])
    plt.show()


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == "__main__":
    images = load_images()

    for img in images:
        frameImage = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        frameImage[0] = frameImage[-1] = 255
        frameImage[1:-1, 0] = frameImage[1:-1, -1] = 255

        borderPixelsImage = cv2.bitwise_and(img, frameImage)

        show_plot(
            [img, frameImage, borderPixelsImage],
            ["Original", "Frame", "Border pixels"],
            cmap=True,
            method=1,
        )

        bordersOnlyImage = morphological_reconstruction(marker=borderPixelsImage, mask=img)

        centralPixelsImage = cv2.bitwise_xor(img, bordersOnlyImage)
        show_plot(
            [img, bordersOnlyImage, centralPixelsImage],
            ["Original", "Border elements only", "Central pixels only"],
            cmap=True,
            method=1,
        )
