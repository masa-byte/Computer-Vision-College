import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_plot(images, titles, cmap=True, method=1):
    plt.figure(figsize=(10, 10))
    if method == 1:
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            plt.imshow(images[i], cmap='gray' if cmap else None)
            plt.title(titles[i])
    else:
        for i in range(len(images)):
            plt.subplot(2, len(images)//2, i+1)
            plt.imshow(images[i], cmap='gray' if cmap else None)
            plt.title(titles[i])
    plt.show()

def load_image():
    folder = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(folder):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

if __name__ == "__main__":
    img = load_image()
    img_out = img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_saturation = img_hsv[:, :, 1]

    img_thresh = cv2.threshold(img_saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    img_diluted = cv2.dilate(img_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    img_reconstructed = morphological_reconstruction(marker=img_open, mask=img_diluted)
    
    x, y, w, h = cv2.boundingRect(img_reconstructed)
    cv2.rectangle(img_out, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)

    show_plot([img, img_saturation], ["Original", "Saturation"], cmap=True)
    show_plot([img_thresh, img_open, img_diluted, img_reconstructed], 
              ["Threshold Otsu", "Opening (marker)", "Diluted (mask)", "Reconstructed"], cmap=True, method=2)
    show_plot([img_out, img_reconstructed], ["Output", "Mask"], cmap=True)
