import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


# ID: 206262123


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    disp_map = np.zeros(img_l.shape)
    disp_min = disp_range[0]
    disp_max = disp_range[1]
    start = k_size
    h, w = img_l.shape

    for i in range(start, h - start):

        for j in range(start, w - start):

            window_l = img_l[i - start:i + start + 1, j - start:j + start + 1]  # left window
            window_l = np.asarray(window_l)
            min = 9999
            disparity = 0

            # check right shift
            for k in range(j + disp_min, j + disp_max):

                if k + start + 1 > w or k - start < 0:
                    break

                window_r = img_r[i - start:i + start + 1, k - start:k + start + 1]  # right window
                window_r = np.asarray(window_r)

                func = ((window_l - window_r) ** 2).sum() # ssd

                if func < min:
                    min = func
                    disparity = k-j

            # check left shift
            for k in range(j - disp_min, j - disp_max,-1):

                if k + start + 1 > w or k - start < 0:
                    break

                window_r = img_r[i - start:i + start + 1, k - start:k + start + 1]  # right window
                window_r = np.asarray(window_r)

                func = ((window_l - window_r) ** 2).sum() # ssd

                if func < min:
                    min = func
                    disparity = abs(k-j)

            disp_map[i][j] = disparity

    return disp_map

    pass


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
# func = ((window_l * window_r).sum()) / math.sqrt(((img_l)**2).sum() * ((img_r)**2).sum())
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """



    disp_map = np.zeros(img_l.shape)
    disp_min = disp_range[0]
    disp_max = disp_range[1]
    start = k_size
    h, w = img_l.shape

    for i in range(start, h - start):

        for j in range(start, w - start):

            window_l = img_l[i - start:i + start + 1, j - start:j + start + 1]  # left window
            window_l = np.asarray(window_l)
            max = 0
            disparity = 0

            # check right shift
            for k in range(j + disp_min, j + disp_max):

                if k + start + 1 > w or k - start < 0:
                    break

                window_r = img_r[i - start:i + start + 1, k - start:k + start + 1]  # right window
                window_r = np.asarray(window_r)

                func = ((window_l * window_r).sum()) / (math.sqrt(((img_l) ** 2).sum() * ((img_r) ** 2).sum())) #nc

                if func > max:
                    max = func
                    disparity = k-j

            # check left shift
            for k in range(j - disp_min, j - disp_max,-1):

                if k + start + 1 > w or k - start < 0:
                    break

                window_r = img_r[i - start:i + start + 1, k - start:k + start + 1]  # right window
                window_r = np.asarray(window_r)

                func = ((window_l * window_r).sum()) / (math.sqrt(((img_l) ** 2).sum() * ((img_r) ** 2).sum())) #nc

                if func > max:
                    max = func
                    disparity = abs(k-j)

            disp_map[i][j] = disparity

    return disp_map

    pass


def Homogeneous(points):
    rows = points.shape[0]
    H_points = np.zeros((rows, 3))
    for i in range(rows):
        for j in range(3):
            if j == 2:
                H_points[i][j] = 1
            else:
                H_points[i][j] = points[i][j]
    return H_points


def unHomogeneous(points):
    return points[:, :-1]


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """

    n = src_pnt.shape[0]

    A = np.zeros((n * 2, 9))

    for i in range(n):
        A[i * 2] = [src_pnt[i][0], src_pnt[i][1], 1, 0, 0, 0, -dst_pnt[i][0] * src_pnt[i][0],
                    -dst_pnt[i][0] * src_pnt[i][1], -dst_pnt[i][0]]

        A[i * 2 + 1] = [0, 0, 0, src_pnt[i][0], src_pnt[i][1], 1, -dst_pnt[i][1] * src_pnt[i][0],
                        -src_pnt[i][1] * dst_pnt[i][1], -dst_pnt[i][1]]

    U, S, V = np.linalg.svd(A)

    H = V[8].reshape((3, 3))

    H = (1 / H.item(8)) * H  # normalize

    # error calc
    error = 0
    src_h = Homogeneous(src_pnt)
    for i in range(src_pnt.shape[0]):
        point = src_h[i]
        point = point.reshape(3, 1)
        pred = H.dot(point)
        pred = pred / pred.item(2)  # normalize
        error = error + abs(unHomogeneous(pred.T).sum() - dst_pnt[i].sum())

    return H, error

    pass


def transform(src_pts, H):
    src = Homogeneous(src_pts)
    pts = np.dot(H, src.T).T
    # normalize
    pts = (pts / pts[:, 2].reshape(-1, 1))[:, 0:2]
    return pts


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    src_p = []
    fig2 = plt.figure()

    # display image 2
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    H, _ = computeHomography(src_p, dst_p)

    h, w = dst_img.shape[:2]

    idx_pts = np.mgrid[0:w, 0:h].reshape(2, -1).T
    map_pts = transform(idx_pts, np.linalg.inv(H))
    map_pts = map_pts.reshape(w, h, 2).astype(np.float32)
    warped = cv2.remap(src_img, map_pts, None, cv2.INTER_CUBIC).T

    mask = warped == 0
    canvas = dst_img * mask + (1 - mask) * warped

    plt.imshow(canvas)
    plt.show()

    pass
