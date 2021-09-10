import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

# id: 206262123

def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    img_h, img_w = in_image.shape[:2]
    kernel_shape = np.array([x for x in kernel.shape])
    mid_ker = kernel_shape // 2
    padded_signal = np.pad(in_image.astype(np.float32),
                           ((kernel_shape[0], kernel_shape[0]),
                            (kernel_shape[1], kernel_shape[1]))
                           , 'edge')

    out_signal = np.zeros_like(in_image)
    for i in range(img_h):
        for j in range(img_w):
            st_x = j + mid_ker[1] + 1
            end_x = st_x + kernel_shape[1]
            st_y = i + mid_ker[0] + 1
            end_y = st_y + kernel_shape[0]

            out_signal[i, j] = (padded_signal[st_y:end_y, st_x:end_x] * kernel).sum()

    return out_signal

def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[1, 0, -1]])
    x_drive = conv2D(in_image, kernel)
    y_drive = conv2D(in_image, kernel.T)

    return x_drive, y_drive


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    Ix,Iy = convDerivative(im2)
    It = im2 - im1

    start = win_size/2
    start = int(start)
    mid = start + 1

    ATA = np.zeros((2,2))
    ATb = np.zeros((2,1))

    XY = np.zeros(2)
    UV = np.zeros(2)

    XY_values = []
    UV_values = []


    for i in range (start , im2.shape[0] , step_size):

        if (i + start + 1 > im2.shape[0]):  # if we reached the border
            break

        for j in range (start, im2.shape[1], step_size):

            if( j + start + 1 > im2.shape[1]): # if we reached the border
                break

            #create windows around points
            window_Ix = Ix[i-mid+1:i+mid, j-mid+1:j+mid].flatten()
            window_Iy = Iy[i-mid+1:i+mid, j-mid+1:j+mid].flatten()
            window_It = It[i-mid+1:i+mid, j-mid+1:j+mid].flatten()

            # formulas
            ATA[0,0] = np.sum(window_Ix * window_Ix)
            ATA[0,1] = np.sum(window_Ix * window_Iy)
            ATA[1,0] = np.sum(window_Iy * window_Ix)
            ATA[1,1] = np.sum(window_Iy * window_Iy)

            ATb[0,0] = np.sum(window_Ix * window_It)*(-1)
            ATb[1,0] = np.sum(window_Iy * window_It)*(-1)

            ATA_inv = np.linalg.inv(ATA)

            e_values = np.linalg.eig(ATA)[0]

            # eigenvalues conditions
            if e_values[1]>1 and (e_values[0]/e_values[1])<100:
                UV_np = np.matmul(ATA_inv,ATb)

                UV[0] = UV_np[1,0]
                UV[1] = UV_np[0,0]

                XY[0] = j
                XY[1] = i

                XY_values.append(XY.copy())
                UV_values.append(UV.copy())

            XY_values.append(XY.copy())
            UV_values.append(UV.copy())


    XY_values = np.asarray(XY_values)
    UV_values = np.asarray(UV_values)

    return XY_values,UV_values

    pass

def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    lap_pyr = []
    for i in range(levels):
        blur_img = cv2.GaussianBlur(img, (5, 5), 0) #blur
        new_img = img - blur_img
        lap_pyr.append(new_img)
        img = blur_img[::2, ::2] #reduce
        if i == levels - 1:
            lap_pyr.append(img) # if we reach last image

    return lap_pyr


pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    """ reconstruction of an image from its Laplacian Pyramid."""


    result = lap_pyr[-1]
    for i in reversed(range(0, len(lap_pyr) - 1)): # go from last to first
        shape = list(result.shape)
        shape[0] *= 2
        shape[1] *= 2
        high_res = np.zeros(shape)
        high_res[::2, ::2] = result
        img_blur = 4*cv2.GaussianBlur(high_res, (5,5), 0) # upsample
        if result.shape[0] * 2 > lap_pyr[i].shape[0]:
            img_blur = np.delete(img_blur, -1, axis=0)
        if result.shape[1] * 2 > lap_pyr[i].shape[1]:
            img_blur = np.delete(img_blur, -1, axis=1)
        result = img_blur + lap_pyr[i]
    return result
    pass


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    gaus_pyr = [img]
    for i in range(levels):
        blur_img = cv2.GaussianBlur(gaus_pyr[i], (5, 5), 0)
        new_img = blur_img[::2, ::2] #reduce
        gaus_pyr.append(new_img)

    return gaus_pyr

pass


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    size = (img.shape[0] * 2, img.shape[1] * 2)
    expanded_im = np.zeros(size)
    expanded_im[::2, ::2] = img
    return conv2D(expanded_im,gs_k) #return expanded_im


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """


    naive_blend = mask * img_1 + (1 - mask) * img_2

    lap_pyr1 = laplaceianReduce(img_1,levels)
    lap_pyr2 = laplaceianReduce(img_2,levels)
    mask_pyr = gaussianPyr(mask,levels)

    assert len(lap_pyr1) == len(lap_pyr2), 'pyramid level are not equal'
    assert len(lap_pyr1) == len(mask_pyr), 'pyramid level are not equal'
    blended_pyramid = []
    for i in range(len(lap_pyr1)):
        blend = mask_pyr[i] * lap_pyr1[i] + (1 - mask_pyr[i]) * lap_pyr2[i] #forumla
        blended_pyramid.append(blend)

    return naive_blend ,laplaceianExpand(blended_pyramid)
    pass
