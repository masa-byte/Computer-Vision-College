import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# Read left and right image (in grayscale)
folder = os.path.dirname(os.path.abspath(__file__))
imgL = cv.imread(folder + '\img1_l.png', 0)
imgR = cv.imread(folder + '\img1_r.png', 0)

# Compute stereo correspondence using the block matching algorithm
stereoBM = cv.StereoBM_create(numDisparities=16, blockSize=11)
disparityBM = stereoBM.compute(imgL, imgR)

uniqueBM, countsBM = np.unique(disparityBM, return_counts=True)
print('stereoBM disparity')
print(len(uniqueBM), dict(zip(uniqueBM, countsBM)))

# Compute stereo correspondence using the modified H. Hirschmuller algorithm
# H. Hirschmuller, 2008. Stereo Processing by Semi-Global Matching and Mutual Information,
# https://core.ac.uk/download/pdf/11134866.pdf
stereoSGBM = cv.StereoSGBM_create(numDisparities=16, blockSize=11)
disparitySGBM = stereoSGBM.compute(imgL, imgR)

uniqueSGBM, countsSGBM = np.unique(disparitySGBM, return_counts=True)
print('stereoSGBM disparity')
print(len(uniqueSGBM), dict(zip(uniqueSGBM, countsSGBM)))

# Display results
plt.subplot(131), plt.imshow(imgL, 'gray')
plt.subplot(132), plt.imshow(disparityBM, 'gray')
plt.subplot(133), plt.imshow(disparitySGBM, 'gray')
plt.show()
