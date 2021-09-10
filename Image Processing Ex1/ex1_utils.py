"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt


from numpy import asarray


LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206262123


def NormalizeData1Bit(data):  # [0,1]
    """
        Convert image from [0,255] representation to [0,1]
        :param data: image with original values
        :return: image with new values
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def NormalizeData8Bit(data):  # [0,255]
    """
            Convert image from [0,1] representation to [0,255]
            :param data: image with original values
            :return: image with new values
    """
    return data * 255


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    if representation == 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        numpy_data = NormalizeData1Bit(asarray(gray_image, dtype=float))
        return numpy_data

    elif representation == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        numpy_data = NormalizeData1Bit(asarray(rgb_image, dtype=float))
        return numpy_data
    else:
        print("No such representation")
        return
    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    numpy_data = imReadAndConvert(filename, representation)  # numpy representation of converted image
    plt.imshow(numpy_data, cmap='gray')
    plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    rgb_to_yiq_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.331]])

    return np.dot(imgRGB, rgb_to_yiq_matrix.T.copy())

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    rgb_to_yiq_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.331]])

    return np.dot(imgYIQ, np.linalg.inv(rgb_to_yiq_matrix).T.copy())

    pass


def channelEQ(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
            Equalizes the histogram of a channel
            :param img: Original Histogram
            :return: Eq'd channel, original histogram, new histogram
    """
    histOrg, bins = np.histogram(img.flatten(), 256)
    cdf = histOrg.cumsum()
    img_Eq = np.interp(img, bins[:-1], cdf)
    histEQ, base = np.histogram(img_Eq.flatten(), 256) # histogram of the eq'd image
    img_Eq = NormalizeData1Bit(img_Eq) # convert back to [0,1]

    return img_Eq, histOrg, histEQ


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: Eq'd image, original histogram, new histogram
    """
    if len(imgOrig.shape) == 3:  # if the image is rgb
        imgOrig = transformRGB2YIQ(imgOrig)

    img_norm = NormalizeData8Bit(imgOrig)  # [0,255]

    if len(imgOrig.shape) == 3:
        Y_channel = img_norm[:, :, 0]  # we change only the Y channel
        Y_channel_Eq, histOrg, histEQ = channelEQ(Y_channel)
        imgOrig[:, :, 0] = Y_channel_Eq
        imgEq = imgOrig
    else:
        imgEq, histOrg, histEQ = channelEQ(img_norm)

    if len(imgOrig.shape) == 3:  # if rgb, we change back to rgb from yiq
        imgEq = transformYIQ2RGB(imgEq)

    return imgEq, histOrg, histEQ

    pass


# def quantizeChannel(imOrig, nQuant, nIter, hist, bins, z, q, img_list, error_list) -> (
#  List[np.ndarray], List[float]):
#     """
#         Quantized a channel in to **nQuant** colors
#         All of the parametrs are from quantizeImage function with same names
#         :return: (List[qImage_i],List[error_i])
#     """
#
#     for it in range(nIter):
#
#
#
#     return img_list, error_list


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 3:
        imOrig = transformRGB2YIQ(imOrig)
        Y_channel = imOrig[:, :, 0]





    if len(imOrig.shape) == 3:
        hist, bins = np.histogram(Y_channel.flatten(), 256) #if rgb picture we work only on Y channel
    else:
        hist, bins = np.histogram(imOrig.flatten(), 256)


    if len(imOrig.shape) == 3:
        img_list, error_list = quantizeChannel(Y_channel, nQuant, nIter, hist, bins, z, q, img_list, error_list)
    else:
        img_list, error_list = quantizeChannel(imOrig, nQuant, nIter, hist, bins, z, q, img_list, error_list)


    plt.plot(error_list)
    plt.show()

    return img_list, error_list

    pass
