import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_and_convert_to_other_color_space(image_path, color_space):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, color_space)

def show_image_with_matplotlib(image, gray):
    if gray:
        # if cmap is set to gray then the image will be shown in grayscale
        plt.imshow(image, cmap='gray')  
    else:
        # if not then the image will be shown in yellow and blue
        # yellow will have the pixel with the highest value, blue lowest, and the rest in between
        # useful when the values of pixels are close and everything looks the same on the image
        plt.imshow(image)  
    plt.show()

def show_image_with_opencv(image):
    # first argument is the name of the window, second is the image
    cv2.imshow("IMAGE", image)
    cv2.waitKey(0)

def save_image(image: np.ndarray, save_path: str):
    cv2.imwrite(save_path, image)


if __name__ == '__main__':
    lena = read_and_convert_to_other_color_space("Vezbe/lena_rgb.png", cv2.COLOR_BGR2RGBA)
    show_image_with_matplotlib(lena, True)
    #show_image_with_opencv(lena)
    save_image(lena, "Vezbe/lena_gray.png")  
    # even though image is in grayscale, it will be saved as RGB
    # so it will have 3 channels with the same value