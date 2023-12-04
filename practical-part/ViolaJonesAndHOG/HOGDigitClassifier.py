"""
https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
"""
import cv2 as cv
import numpy as np
import itertools as it

SZ = 20  # width and height of single digit in pixels
CLASS_N = 10  # number of digit classes


def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    output = it.zip_longest(fillvalue=fillvalue, *args)
    return output


def mosaic(w, imgs):
    """Make a grid from images.
    w    -- number of grid columns
    imgs -- images (must have same size and format)
    """
    imgs = iter(imgs)
    img0 = next(imgs)    
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def load_digits(fn):
    digits_img = cv.imread(fn, 0)
    digits = split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels


def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img


def svmInit(C=12.5, gamma=0.50625):
    model = cv.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv.ml.SVM_RBF)
    model.setType(cv.ml.SVM_C_SVC)  
    return model


def svmTrain(model, samples, responses):
    model.train(samples, cv.ml.ROW_SAMPLE, responses)
    return model


def svmPredict(model, samples):
    return model.predict(samples)[1].ravel()


def svmEvaluate(model, digits, samples, labels):
    predictions = svmPredict(model, samples)
    accuracy = (labels == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, predictions == labels):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0
        vis.append(img)
        
    return mosaic(25, vis)


def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog():
    winSize = (20, 20)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                           histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return hog    


print('Loading digits from digits.png ... ')
digits, labels = load_digits('digits.png')

print('Shuffle data ... ')
rand = np.random.RandomState(10)
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

print('Deskew images ... ')
digits_deskewed = list(map(deskew, digits))

print('Defining HoG parameters ...')
hog = get_hog()

print('Calculating HoG descriptor for every image ... ')
hog_descriptors = []
for img in digits_deskewed:
    hog_descriptors.append(hog.compute(img))
hog_descriptors = np.squeeze(hog_descriptors)

print('Splitting data into training (90%) and test set (10%)... ')
train_n = int(0.9 * len(hog_descriptors))
digits_train, digits_test = np.split(digits_deskewed, [train_n])
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
labels_train, labels_test = np.split(labels, [train_n])
    
print('Training SVM model ...')
model = svmInit()
svmTrain(model, hog_descriptors_train, labels_train)

print('Evaluating model ... ')
vis = svmEvaluate(model, digits_test, hog_descriptors_test, labels_test)

cv.imshow("Output", vis)
cv.waitKey(0)
cv.destroyAllWindows()
