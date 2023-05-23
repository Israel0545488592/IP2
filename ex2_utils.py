from scipy.stats import norm
import numpy as np
import cv2
from itertools import product
from typing import Tuple

LAPLUS_KER = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])



def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 212403679


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    res = np.zeros(len(in_signal) + len(k_size) - 1)

    for res_ind, ker_ind in product(range(len(res)), range(len(k_size))):
        if 0 <= res_ind - ker_ind < len(in_signal):
            res[res_ind] += k_size[ker_ind] * in_signal[res_ind - ker_ind]

    return res


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    if len(kernel.shape) < 2: kernel = kernel.reshape(1, len(kernel))
    row_pad, col_pad = kernel.shape[0] // 2, kernel.shape[1] // 2

    pad_img = np.pad(in_image, ((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])), 'reflect')
    res = np.zeros_like(in_image)

    for res_row, res_col in product(range(in_image.shape[0]), range(in_image.shape[1])):
        for ker_row, ker_col in product(range(kernel.shape[0]), range(kernel.shape[1])):
            res[res_row, res_col] += kernel[ker_row, ker_col] * pad_img[res_row + ker_row + row_pad + 1][res_col + ker_col + col_pad + 1]

    return res


def convDerivative(in_image: np.ndarray) -> Tuple[np.ndarray]:
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    col_drv = cv2.filter2D(in_image, -1, np.array([-1, 0, 1]).reshape(3, 1))
    row_drv = cv2.filter2D(in_image, -1, np.array([-1, 0, 1]))

    grad_mag_trans = np.vectorize(lambda x, y : np.sqrt(x * x + y * y))
    grad_dir_trans = np.vectorize(lambda x, y : np.arctan(y / x) if x != 0 else np.pi / 2)

    return grad_dir_trans(row_drv, col_drv), grad_mag_trans(row_drv, col_drv)


def gaus_ker(size: int, std: int) -> np.ndarray:

    ker = np.diff(norm.cdf(np.linspace(-size * std, size * std, size + 1)))
    return ker.reshape(size, 1) / ker.sum()


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return conv2D(in_image, gaus_ker(k_size, 0.2))


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return cv2.filter2D(in_image, -1, cv2.getGaussianKernel(k_size, 0.2), borderType = cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return np.where(np.abs(cv2.filter2D(img, -1, LAPLUS_KER)) < 0.01, 1, 0)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    LoG_filtered_img = cv2.filter2D(cv2.filter2D(img, -1, LAPLUS_KER), -1, cv2.getGaussianKernel(3, 0.2))

    return np.where(np.abs(LoG_filtered_img) < 0.1, 1, 0)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> Tuple[np.ndarray]:
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return