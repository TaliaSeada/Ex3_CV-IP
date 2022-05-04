import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211551601


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # create the returned arrays
    x_y = []
    u_v = []

    # convert to GRAY scale
    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        plt.gray()

    # Compute the gradients I_x and I_y
    ker = np.array([[1, 0, -1]])
    Ix = cv2.filter2D(im2, -1, ker, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, ker.T, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    for i in range(0, im1.shape[0] - win_size, step_size):
        for j in range(0, im1.shape[1] - win_size, step_size):
            # take the window
            Ix_cut = Ix[i:i + win_size, j:j + win_size].flatten()
            Iy_cut = Iy[i:i + win_size, j:j + win_size].flatten()
            It_cut = It[i:i + win_size, j:j + win_size].flatten()

            # create A and b
            A = np.array([Ix_cut, Iy_cut]).T
            b = It_cut.T

            # calculate v
            AtA = A.T.dot(A)
            AtA_inv = np.linalg.inv(AtA)
            u, v = AtA_inv.dot(A.T.dot(b))

            # calculate the eigenvalues
            e1, e2 = np.linalg.eigvals(AtA)
            eig_max = max(e1, e2)
            eig_min = min(e1, e2)

            if eig_max / eig_min < 100 and eig_min > 1:
                x_y.append([j, i])
                u_v.append([u, v])

    return np.array(x_y), np.array(u_v)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    im1_pyramids = gaussianPyr(img1, k)
    im2_pyramids = gaussianPyr(img2, k)

    p = []
    v = []
    points, vectors = opticalFlow(im1_pyramids[-1], im2_pyramids[-1], stepSize, winSize)
    p.append(points)
    v.append(vectors)

    im1_pyramids.reverse()
    im2_pyramids.reverse()

    for i in range(k):
        points, vectors = opticalFlow(im1_pyramids[i], im2_pyramids[i], stepSize, winSize)
        p.append(points)
        v.append(vectors)

        U = points * 2 + p[i-1]
        V = vectors * 2 + v[i-1]

    return np.ndarray(U, V, 2)


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyrs = [img]
    k_size = 5
    sigma = 0.3*((k_size-1)*0.5-1) + 0.8
    gauss_ker = cv2.getGaussianKernel(k_size, sigma)

    for i in range(levels):
        img = cv2.filter2D(img, -1, gauss_ker, borderType=cv2.BORDER_REPLICATE)
        img = img[::2, ::2]
        pyrs.append(img)

    return pyrs


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyrs = [img]
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    gauss_ker = cv2.getGaussianKernel(k_size, sigma)
    gaussPyr = gaussianPyr(img, levels)

    for i in range(levels):
        img = cv2.filter2D(img, -1, gauss_ker, borderType=cv2.BORDER_REPLICATE)
        img = img[::2, ::2]
        pyrs.append(img)

    return pyrs


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass
