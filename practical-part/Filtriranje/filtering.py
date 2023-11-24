import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    # fft2 is fast fourier transform in 2 dimensions
    img_fft = np.fft.fft2(img) 
    # this function shifts the zero-frequency component to the center of the spectrum
    # meaning low frequencies are in the center and high frequencies are on the edges
    img_fft = np.fft.fftshift(img_fft)
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1):
    # we need to undo log to get the amplitude of the frequency domain
    # we need to multiply the amplitude with the complex number to get the complex number with the amplitude
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    # iff2 is inverse fast fourier transform in 2 dimensions
    # it will return the image from frequency domain to spatial domain
    # result of iff2 is a complex image, but we are only interested in module
    img_filtered = np.abs(np.fft.ifft2(img_fft)) 
    return img_filtered


def fft_noise_addition(img, center):
    img_fft = fft(img)
    # image in frequency domain is a complex number, we need the amplitude of that complex number (which is the module)
    img_fft_mag = np.abs(img_fft) 

    # we need to divide the complex number with its amplitude to get the complex number with amplitude 1
    img_mag_1 = img_fft / img_fft_mag 

    # these values are too big to be changed directly, so we use log to visualize the amplitude of the frequency domain
    img_fft_log = np.log(img_fft_mag)
    plt.subplot(1, 2, 1)
    plt.imshow(img_fft_log)
    img_fft_log[center[0] - 50, center[1] - 50] = 16
    img_fft_log[center[0] + 50, center[1] + 50] = 16
    img_filtered = inverse_fft(img_fft_log, img_mag_1)
    plt.subplot(1, 2, 2)
    plt.imshow(img_fft_log)
    plt.show()

    return img_filtered

def low_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # this is the equation of a circle, we want everything outside the circle to be zero 
            # that represents ideal low pass filter
            if (x-center[0])**2 + (y-center[1])**2 > radius*radius:
                img_fft_log[x,y] = 0

    plt.subplot(1, 2, 1)
    plt.imshow(img_fft_log)

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered

def high_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # this is the equation of a circle, we want everything inside of the circle to be zero
            # that represents ideal high pass filter
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) < radius*radius: 
                img_fft_log[x,y] = 0

    plt.subplot(1, 2, 2)
    plt.imshow(img_fft_log)

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered


def filtering_in_spatial_domain(img):
    gaus_kernel = cv2.getGaussianKernel(ksize=21, sigma=7)
    custom_kernel = np.zeros((3, 3), dtype=np.int8)
    # custom kernel becomes [[0 0 0] [0 0 1] [0 0 0]]
    custom_kernel[1, 2] = 2
    img_gauss_blur = cv2.filter2D(img, -1, gaus_kernel) 
    img_filter_custom1 = cv2.filter2D(img, -1, custom_kernel)
    custom_kernel[1, 2] = 5
    img_filter_custom2 = cv2.filter2D(img, -1, custom_kernel)

    return img_gauss_blur, img_filter_custom1, img_filter_custom2


if __name__ == '__main__':
    # image will be loaded in BGR format
    img = cv2.imread("Vezbe/lena_rgb.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = img.shape

    # image is (512, 512) so the center is (256, 256)
    center = (int(shape[0]/2), int(shape[1]/2)) 
    # radius of ideal low or high pass filter
    radius = 50

    # filtering in spatial domain with gaussian and custom kernel
    img_gauss, img_custom1, img_custom2 = filtering_in_spatial_domain(img)

    # plt.figure(1)
    # plt.imshow(img_gauss, cmap='gray')
    # #plt.show(block=False)

    # plt.figure(2)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_custom1, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_custom2, cmap='gray')
    # plt.show()

    # adding periodic noise to image in frequency domain
    img_noise_added = fft_noise_addition(img, center)

    plt.figure()
    img_low_pass = low_pass_filter(img, center, radius)
    img_high_pass = high_pass_filter(img, center, radius)
    plt.show()

    plt.figure(figsize=(50, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img_noise_added, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img_low_pass, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img_high_pass, cmap='gray')
    plt.show()


