import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def show_plot(images, titles, cmap=True):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='gray' if cmap else None)
        plt.title(titles[i])
    plt.show()


def load_images():
    folder = os.path.dirname(os.path.abspath(__file__))
    images = []
    print(folder)
    for file in os.listdir(folder):
        if file.endswith('.png') and '3' in file:
            image_path = os.path.join(folder, file)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    return images


def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft


def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))
    return img_filtered


def calc_mean(img_fft_log, x, y):
    sum = 0
    count = 0
    radius = 2
    for i in range(x-radius, x+radius):
        for j in range(y-radius, y+radius):
            if i >= 0 and i < img_fft_log.shape[0] and j >= 0 and j < img_fft_log.shape[1]:
                sum += img_fft_log[i][j]
                count += 1
    return sum / count


def remove_noise(img, sharp_image):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)

    sharp_image_fft = fft(sharp_image)
    sharp_image_fft_mag = np.abs(sharp_image_fft)
    sharp_image_fft_log = np.log(sharp_image_fft_mag)

    difference = img_fft_log - sharp_image_fft_log

    show_plot([img_fft_log, sharp_image_fft_log, difference], [
        'Original image frequencies', 'Edgies frequencies', 'Difference'])

    values = {}
    black_pixels = difference < 0
    for x in range(black_pixels.shape[0]):
        for y in range(black_pixels.shape[1]):
            if black_pixels[x][y]:
                values[(x, y)] = img_fft_log[x][y]

    mean = np.mean(list(values.values()))
    std = np.std(list(values.values()))
    for key, value in values.items():
        if value - mean > 5 * std:
            img_fft_log[key[0]][key[1]] = calc_mean(
                img_fft_log, key[0], key[1])

    new_img_fft = img_mag_1 * np.exp(img_fft_log)
    return np.abs(np.fft.ifft2(new_img_fft))


def noise_reduction(image):
    # img_sharpened = cv2.Sobel(image, -1, 1, 1, ksize=5)
    img_sharpened = cv2.Laplacian(image, -1, ksize=5)
    img_no_noise = remove_noise(image, img_sharpened)
    show_plot([image, img_sharpened], ['Original image', 'Sharpened image'])
    return img_no_noise


if __name__ == '__main__':
    images = load_images()
    for image in images:
        img_no_noise = noise_reduction(image)
        show_plot([image, img_no_noise], ['Original image', 'No noise'])
