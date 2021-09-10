import numpy as np
import cv2
from collections import defaultdict

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


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """

    result = []
    len_result = len(inSignal) + len(kernel1) - 1  # Size of convolution

    for i in range(len_result):
        sum = 0
        for j in range(len(inSignal)):
            if i - j >= 0 and len(kernel1) > i - j:  # There is no overlap
                sum += inSignal[j] * kernel1[i - j]  # Convolution's definition
        result.append(sum)

    return result


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    1
    :return: The convolved image
    """
    kernel = np.flip(kernel2)  # Flip the kernel
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_padded = np.pad(inImage, (kh // 2, kw // 2), mode='edge')  # Padding
    output = np.ndarray(inImage.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.multiply(image_padded[i:i + kh, j:j + kw], kernel).sum()

    return output



def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    # derivative matrix (x)
    x = np.array(([0, 0, 0],
                  [1, 0, -1],
                  [0, 0, 0],
                  ))
    # derivative matrix (y)
    y = np.transpose(x)

    Gx = cv2.filter2D(src=inImage, kernel=x, ddepth=-1, borderType=cv2.BORDER_REPLICATE)
    Gy = cv2.filter2D(src=inImage, kernel=y, ddepth=-1, borderType=cv2.BORDER_REPLICATE)

    magnitude = np.sqrt(Gx ** 2.0 + Gy ** 2.0)
    angle = np.arctan2(Gy, Gx)  # direction matrix

    return magnitude, angle, Gx, Gy


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
    gaussian = cv2.getGaussianKernel(ksize=kernel_size[0], sigma=sigma)
    output = cv2.filter2D(src=in_image, kernel=gaussian, ddepth=-1, borderType=cv2.BORDER_REPLICATE)
    return output


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """

    # my implementation

    # 1/8 for correct magnitude
    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (1 / 8)
    y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (1 / 8)

    sob_x = conv2D(img, x)
    sob_y = conv2D(img, y)

    my_mag = np.sqrt(sob_x ** 2 + sob_y ** 2)

    # binary representation based on threshold
    my_mag[my_mag >= thresh] = 1
    my_mag[my_mag < thresh] = 0

    # cv implementation

    sob_x = cv2.Sobel(img, cv2.CV_64F, 1, 0) / (1 / 8)
    sob_y = cv2.Sobel(img, cv2.CV_64F, 0, 1) / (1 / 8)

    cv_mag = np.sqrt(sob_x ** 2 + sob_y ** 2)

    cv_mag[cv_mag >= thresh] = 1
    cv_mag[cv_mag < thresh] = 0

    return cv_mag, my_mag


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """

    # gaussian smoothing
    blur_img = blurImage2(img, np.array([5, 5]))

    der = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

    d = cv2.filter2D(src=blur_img, kernel=der, ddepth=-1, borderType=cv2.BORDER_REPLICATE)

    output = np.zeros(d.shape)

    # zero crossing - check if there are changes in signs between nearby pixels
    for i in range(1, d.shape[0] - 1):
        for j in range(1, d.shape[1] - 1):
            count = 0
            if (((d[i - 1, j] > 0 and d[i, j] < 0) or (d[i - 1, j] < 0 and d[i, j] > 0)) or (
                    (d[i, j - 1] > 0 and d[i, j] < 0) or (d[i, j - 1] < 0 and d[i, j] > 0))):
                count += 1
            elif (((d[i + 1, j] > 0 and d[i, j] < 0) or (d[i + 1, j] < 0 and d[i, j] > 0)) or (
                    (d[i, j + 1] > 0 and d[i, j] < 0) or (d[i, j + 1] < 0 and d[i, j] > 0))):
                count += 1

            if count > 0:
                output[i, j] = 1

    return output


def non_maximum_suppression(image, angles):
    """
        get a better estimate of the magnitude values of the pixels in the gradient direction
        :param imgage: magnitude
        :param angles: angles
        :return: suppressed image
    """
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif 22.5 <= angles[i, j] < 67.5:
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif 67.5 <= angles[i, j] < 112.5:
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())  # normalize
    return suppressed


def hysteresis(suppressed, low_t, high_t):
    """
            Any pixel above the upper threshold is turned white. The surround pixels are then searched recursively.
             If the values are greater than the lower threshold they are also turned white.
            :param suppressed:
            :param low_t: low threshold
            :param high_t high threshold
            :return: thresholded image
    """

    thresholded = np.zeros(suppressed.shape)
    for i in range(0, suppressed.shape[0]):  # loop over pixels
        for j in range(0, suppressed.shape[1]):
            if suppressed[i, j] < low_t:  # lower than low threshold
                thresholded[i, j] = 0
            else:  # higher than high threshold
                thresholded[i, j] = 255
    return thresholded


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """

    img = img * 255

    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0) / (1 / 8)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1) / (1 / 8)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    angles = np.arctan2(Gy, Gx)

    # my implementation

    nms = non_maximum_suppression(magnitude, angles)
    my_canny = hysteresis(nms, thrs_1, thrs_2)

    # cv implementation

    cv_canny = cv2.Canny(img.astype(np.uint8), thrs_1, thrs_2)
    return cv_canny, my_canny


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    img = img * 255
    edge_image = cv2.Canny(img.astype(np.uint8), 100, 200)

    img_h = edge_image.shape[0]
    img_w = edge_image.shape[1]

    # calculate theta range with 100 thetas
    d = int(360 / 100)
    # create bins
    bins = np.arange(0, 360, step=d)
    # radius ranges from min to max with increment of 1
    ranges = np.arange(min_radius, max_radius, step=1)

    cos = np.cos(np.deg2rad(bins))
    sin = np.sin(np.deg2rad(bins))

    circle_candidates = []
    for i in ranges:
        for j in range(100):
            circle_candidates.append((i, int(i * cos[j]), int(i * sin[j])))

    accumulator = defaultdict(int)  # each item represents a circle(placement and radius)

    # accumulate votes for each item
    for i in range(img_h):
        for j in range(img_w):
            if edge_image[i][j] != 0:
                # Add new item and count votes for a certain pixel to be a circle based on edges
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = j - rcos_t
                    y_center = i - rsin_t
                    accumulator[(x_center, y_center, r)] += 1

    out_circles = []

    # Sort accumulator items based on the votes and remove circles with low vote %

    for candidate_circle, votes in sorted(accumulator.items()):
        x, y, r = candidate_circle
        current_vote_percentage = votes / 100
        if current_vote_percentage >= 0.5:  # Only circles above 0.5 threshold
            out_circles.append((x, y, r))

    circles_final = []
    for x1, y1, r1, in out_circles:
        # Remove close circles that represent the same spot
        if all((x1 - x2) ** 2 + (y1 - y2) ** 2 > r2 ** 2 for x2, y2, r2 in circles_final):
            circles_final.append((x1, y1, r1,))

    return circles_final
