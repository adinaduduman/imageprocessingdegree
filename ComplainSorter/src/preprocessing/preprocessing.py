import cv2
import numpy as np
import os
import math
from skimage.filters import threshold_local


def run_preproc(imageName):

    print('Start Preprocessing')
    print('Start adjusting the contrast')

    image = cv2.imread(f'../data/1_raw_complains/{imageName}')
    adjustContrast(image, imageName) # prepare image

    print('Adjusting the contrast is completed')
    print('Start line segmentation')

    image = cv2.imread(f'../data/2_contrast_proc/{imageName}')
    lineSegmentation(image, imageName)

    print('Line segmentation is completed')
    print('Start word segmentation')

    wordSegmentation(imageName, 65)
    print('Word segmentation is completed')
    print('Preprocesing is completed')


def adjustContrast(image, imageName):
    # increase contrast
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(image, 11, offset=10, method="median")
    # # image = cv2.medianBlur(image,3);
    # # image = cv2.bilateralFilter(image, 5, 50, 50)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = (image > T).astype("uint8") * 255

    # write
    cv2.imwrite(f'../data/2_contrast_proc/{imageName}', image)


def lineSegmentation(image, imageName):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        # show ROI
        directory = f'../data/3_line_segmentation/{imageName.split(".")[0]}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        cv2.imwrite(f'../data/3_line_segmentation/{imageName.split(".")[0]}/{str(i)}_{imageName}', roi)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2) # improves word accuracy


def wordSegmentation(imageName, height):
    # read input images from 'in' directory
    directory = f'../data/3_line_segmentation/{imageName.split(".")[0]}'
    imgFiles = os.listdir(directory)

    for (i, f) in enumerate(imgFiles):
        print('Segmenting words of sample %s' % f)

        # resize image
        img = cv2.imread(f'{directory}/%s'%f)
        assert img.ndim in (2, 3)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img.shape[0]
        factor = height / h
        img = cv2.resize(img, dsize=None, fx=factor, fy=factor)

        res = wordSegmentationProcess(img, kernelSize=25, sigma=11, theta=9, minArea=200)

        newDirectory = f'../data/4_word_segmentation/{imageName.split(".")[0]}'
        if not os.path.exists(newDirectory):
            os.mkdir(newDirectory)

        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite(f'{newDirectory}/{i}{j}.{imageName.split(".")[1]}', wordImg) # save word
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)

        # cv2.imwrite(f'{newDirectory}/summary{i}.{imageName.split(".")[1]}', img)


def wordSegmentationProcess(img, kernelSize=25, sigma=11, theta=7, minArea=0):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry: entry[0][0])


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

