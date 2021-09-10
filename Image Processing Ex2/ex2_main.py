from ex2_utils import *
import matplotlib.pyplot as plt
import numpy as np

from numpy import asarray


def conv1Demo():
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 0.5])
    result = conv1D(a, b)

    print('1D convolution')
    print('array', a)
    print('kernel', b)
    print('result', result)
    print('\n')


def conv2Demo():
    img = cv2.imread('flowers.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = NormalizeData1Bit(asarray(img, dtype=float))

    blur = np.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1])) / 9

    result = conv2D(img, blur)
    # I used blur 3 times to make it look more visable
    result = conv2D(result, blur)
    result = conv2D(result, blur)
    result = conv2D(result, blur)

    f, ax = plt.subplots(1, 2)
    ax[0].set_title('original')
    ax[1].set_title('blurred')
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(result, cmap='gray')
    plt.show()


def derivDemo():
    img = cv2.imread('panda.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = NormalizeData1Bit(asarray(img, dtype=float))

    m, d, x, y = convDerivative(img)

    f, ax = plt.subplots(2, 2, figsize=(6, 8))

    ax[0, 0].set_title('Gx')
    ax[0, 1].set_title('Gy')
    ax[1, 0].set_title('direction')
    ax[1, 1].set_title('magnitude')

    ax[0, 0].imshow(x, cmap='gray')
    ax[0, 1].imshow(y, cmap='gray')
    ax[1, 0].imshow(d, cmap='gray')
    ax[1, 1].imshow(m, cmap='gray')

    plt.show()


def blurDemo():
    img = cv2.imread('flowers.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = NormalizeData1Bit(asarray(img, dtype=float))
    result = blurImage2(img, np.array([9, 9]))

    f, ax = plt.subplots(1, 2)
    ax[0].set_title('original')
    ax[1].set_title('gaussian blur')
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(result, cmap='gray')
    plt.show()


def edgeDemo():
    img = cv2.imread('panda.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = NormalizeData1Bit(asarray(img, dtype=float))

    cv_edge, my_edge = edgeDetectionSobel(img)

    edge_zero = edgeDetectionZeroCrossingLOG(img)

    cv_canny, my_canny = edgeDetectionCanny(img, 10, 200)

    f, ax = plt.subplots(2, 3, figsize=(6, 8))

    ax[0, 0].set_title('my sobel')
    ax[0, 1].set_title('cv sobel')
    ax[1, 0].set_title('LOG')
    ax[1, 1].set_title('my canny')
    ax[1, 2].set_title('cv canny')

    ax[0, 0].imshow(my_edge, cmap='gray')
    ax[0, 1].imshow(cv_edge, cmap='gray')
    ax[1, 0].imshow(edge_zero, cmap='gray')
    ax[1, 1].imshow(my_canny, cmap='gray')
    ax[1, 2].imshow(cv_canny, cmap='gray')

    plt.show()


def houghDemo():
    print('Hough Circles')
    img = cv2.imread('circles2.png', 0)
    img = NormalizeData1Bit(img)
    circles = houghCircle(img, 30, 70)
    print('list of circles', circles)


def main():
    print("ID:", myID())

    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


if __name__ == '__main__':
    main()
