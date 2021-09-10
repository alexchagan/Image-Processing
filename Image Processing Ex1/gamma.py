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
from ex1_utils import *
import cv2 as cv
import argparse


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    template taken from https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    img = imReadAndConvert(img_path, rep)

    alpha_slider_max = 200
    title_window = 'Gamma Correction'

    def on_trackbar(val):
        gamma = val/100 # we want value between 0 and 2
        img_gamma1 = np.power(img, gamma).clip(0, 255) # gamma correction
        cv.imshow(title_window, img_gamma1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', help=img_path, default=img_path)
    args = parser.parse_args()

    src1 = cv.imread(cv.samples.findFile(args.input1))

    if src1 is None:
        print('Could not open or find the image: ', args.input1)
        exit(0)

    cv.namedWindow(title_window)

    trackbar_name = 'Gamma x %d' % alpha_slider_max
    cv.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)

    # Show some stuff
    on_trackbar(0)

    # Wait until user press some key
    cv.waitKey()

    pass


def main():
   gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)




if __name__ == '__main__':
    main()
