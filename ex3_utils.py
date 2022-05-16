import math
import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError

# TODO comments
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

    # Compute the gradients I_x and I_y
    ker = np.array([[-1, 0, 1]])
    Ix = cv2.filter2D(im2, -1, ker, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, ker.T, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    w = win_size // 2
    for i in range(w, im1.shape[0] - w + 1, step_size):
        for j in range(w, im1.shape[1] - w + 1, step_size):
            # take the window
            Ix_cut = Ix[i - w:i + w, j - w:j + w].flatten()
            Iy_cut = Iy[i - w:i + w, j - w:j + w].flatten()
            It_cut = It[i - w:i + w, j - w:j + w].flatten()

            AtA = [[(Ix_cut * Ix_cut).sum(), (Ix_cut * Iy_cut).sum()],
                   [(Ix_cut * Iy_cut).sum(), (Iy_cut * Iy_cut).sum()]]

            # calculate the eigenvalues
            e1, e2 = np.linalg.eigvals(AtA)
            eig_max = max(e1, e2)
            eig_min = min(e1, e2)

            if eig_max / eig_min >= 100 or eig_min <= 1:
                continue

            # calculate v
            Atb = [[-(Ix_cut * It_cut).sum()], [-(Iy_cut * It_cut).sum()]]
            u, v = np.linalg.inv(AtA) @ Atb
            x_y.append([j, i])
            u_v.append([u[0], v[0]])

    return np.array(x_y), np.array(u_v)


def expand_LK(img):
    h = img.shape[0] * 2
    w = img.shape[1] * 2
    if len(img.shape) > 2:
        newImg = np.zeros((h, w, 3))
    else:
        newImg = np.zeros((h, w))

    newImg[::2, ::2] = img
    return newImg


# TODO zeros
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

    # expand the image with zeros then fill in the blanks with the last layer image
    im1_pyrs = gaussianPyr(img1, k)
    im1_pyrs.reverse()
    im2_pyrs = gaussianPyr(img2, k)
    im2_pyrs.reverse()

    points_i = []
    vectors_i = []
    for i in range(len(im2_pyrs)):
        points_i, vectors_i = opticalFlow(im1_pyrs[i], im2_pyrs[i], step_size=stepSize, win_size=winSize)
        # if check_zeros(vectors_i):
        #     return vectors_i, points_i

        new_img = expand_LK(im1_pyrs[i])

        for j in range(len(points_i)):
            new_img[points_i[j][0]] += 2 * vectors_i[j][0]
            new_img[points_i[j][1]] += 2 * vectors_i[j][1]

        if i < len(im2_pyrs)-1:
            new_img += im1_pyrs[i+1]
            im1_pyrs[i+1] = new_img/255

    return vectors_i, points_i


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    points, vectors = opticalFlow(im1.astype(np.float), im2.astype(np.float), step_size=20, win_size=5)

    x = []
    y = []
    med_u = np.median(vectors[0])
    med_v = np.median(vectors[1])
    for i in range(len(vectors)):
        if abs(vectors[i][0]) > med_u and abs(vectors[i][1]) > med_v:
            x.append(vectors[i][0])
            y.append(vectors[i][1])

    t_x = np.median(x)*2
    t_y = np.median(y)*2

    mat = np.float32([
        [1, 0, t_x],
        [0, 1, t_y],
        [0, 0, 1]
    ])
    return mat


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    points, vectors = opticalFlow(im1.astype(np.float), im2.astype(np.float), step_size=20, win_size=5)
    first_pnts = []
    second_pnts = [] = []
    for i in range(1, len(points), 5):
        first_pnts.append(points[i])
        point = (int(points[i][0] + vectors[i][0]), int(points[i][1] + vectors[i][1]))
        second_pnts.append(point)

    angs = []
    for i in range(len(first_pnts)):
        ang = find_ang(first_pnts[i], second_pnts[i])
        angs.append(ang)

    theta = (360 - np.median(angs))*2
    x = []
    y = []
    med_u = np.median(vectors[0])
    med_v = np.median(vectors[1])
    for i in range(len(vectors)):
        if abs(vectors[i][0]) > med_u and abs(vectors[i][1]) > med_v:
            x.append(vectors[i][0])
            y.append(vectors[i][1])

    t_x = np.mean(x)
    t_y = np.mean(y)

    mat = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
        [0, 0, 1]
    ])
    return mat


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pad = np.max(im1.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1, pad))
    fft2 = np.fft.fft2(np.pad(im2, pad))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + pad:-pad + 1, 1 + pad:-pad + 1]
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(im2.shape) // 2

    # fig, (ax_img1, ax_img2, ax_corr) = plt.subplots(1, 3, figsize=(15, 5))
    # im = ax_img1.imshow(im1, cmap='gray')
    # ax_img1.set_title('img1')
    # ax_img2.imshow(im2, cmap='gray')
    # ax_img2.set_title('img2')
    # im = ax_corr.imshow(corr, cmap='viridis')
    # ax_corr.set_title('Cross-correlation')
    # ax_img1.plot(x, y, 'ro')
    # ax_img2.plot(x2, y2, 'go')
    # ax_corr.plot(x, y, 'ro')
    # fig.show()

    t_x = x2 - x - 1
    t_y = y2 - y - 1

    mat = np.float32([
        [1, 0, t_x],
        [0, 1, t_y],
        [0, 0, 1]
    ])
    return mat


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pad = np.max(im1.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1, pad))
    fft2 = np.fft.fft2(np.pad(im2, pad))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + pad:-pad + 1, 1 + pad:-pad + 1]
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(im2.shape) // 2

    # fig, (ax_img1, ax_img2, ax_corr) = plt.subplots(1, 3, figsize=(15, 5))
    # im = ax_img1.imshow(im1, cmap='gray')
    # ax_img1.set_title('img1')
    # ax_img2.imshow(im2, cmap='gray')
    # ax_img2.set_title('img2')
    # im = ax_corr.imshow(corr, cmap='viridis')
    # ax_corr.set_title('Cross-correlation')
    # ax_img1.plot(x, y, 'ro')
    # ax_img2.plot(x2, y2, 'go')
    # ax_corr.plot(x, y, 'ro')
    # fig.show()

    theta = find_ang((x2, y2), (x, y))
    mat = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
        [0, 0, 1]
    ])
    mat = np.linalg.inv(mat)
    rotate = cv2.warpPerspective(im2, mat, im2.shape[::-1])

    pad = np.max(im1.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1, pad))
    fft2 = np.fft.fft2(np.pad(rotate, pad))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + pad:-pad + 1, 1 + pad:-pad + 1]
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(rotate.shape) // 2

    t_x = x2 - x - 1
    t_y = y2 - y - 1

    mat = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
        [0, 0, 1]
    ])

    return mat


def find_ang(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    T = np.linalg.inv(T)
    # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    for i in range(im2.shape[0]-1):
        for j in range(im2.shape[1]-1):
            x = (T[0][0] * i + T[0][1] * j + T[0][2]) / (T[2][0] * i + T[2][1] * j + T[2][2])
            y = (T[1][0] * i + T[1][1] * j + T[1][2]) / (T[2][0] * i + T[2][1] * j + T[2][2])
            # calculate the percentage to take from each pixel
            a = x - np.floor(x)
            b = y - np.floor(y)
            # formula from lecture
            if a != 0 or b != 0:
                im2[i][j] = (1 - a) * (1 - b) * im1[i][j] + a * (1 - b) * im1[i + 1][j] + a * b * im1[i + 1][j + 1] + \
                            (1 - a) * b * im1[i][j + 1]
            else:
                im2[i][j] = im1[int(x)][int(y)]

    return im2


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
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    gauss_ker = cv2.getGaussianKernel(k_size, sigma)
    gauss_ker = np.dot(gauss_ker, gauss_ker.T)

    for i in range(1, levels):
        img = cv2.filter2D(img, -1, gauss_ker, borderType=cv2.BORDER_REPLICATE)
        img = img[::2, ::2]
        pyrs.append(img)

    return pyrs


def expand(img):
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    gauss_ker = cv2.getGaussianKernel(k_size, sigma)
    gauss_ker = np.dot(gauss_ker, gauss_ker.T)

    h = img.shape[0] * 2
    w = img.shape[1] * 2
    if len(img.shape) > 2:
        newImg = np.zeros((h, w, 3))
    else:
        newImg = np.zeros((h, w))

    newImg[::2, ::2] = img
    newImg = cv2.filter2D(newImg, -1, gauss_ker * 4, borderType=cv2.BORDER_REPLICATE)
    return newImg


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaussPyr = gaussianPyr(img, levels)
    pyrs = [gaussPyr[-1]]
    for i in range(levels-1, 0, -1):
        expanded = expand(gaussPyr[i])
        lap = cv2.subtract(gaussPyr[i - 1], expanded)
        pyrs.append(lap)
    pyrs.reverse()
    return pyrs


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    restored = [lap_pyr[-1]]
    for i in range(len(lap_pyr)-1, 0, -1):
        exp = expand(restored[-1])
        res = lap_pyr[i-1] + exp
        restored.append(res)

    return restored[-1]


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
    img_1 = cv2.resize(img_1, (mask.shape[1], mask.shape[0]))
    img_2 = cv2.resize(img_2, (mask.shape[1], mask.shape[0]))

    im1_lap = laplaceianReduce(img_1, levels)
    im2_lap = laplaceianReduce(img_2, levels)
    mask_pyr = gaussianPyr(mask, levels)

    lap_for_exp = []
    for i in range(levels):
        merge = im1_lap[i] * mask_pyr[i] + (1 - mask_pyr[i]) * im2_lap[i]
        lap_for_exp.append(merge)
    new_img = laplaceianExpand(lap_for_exp)
    naive = img_1 * mask + (1 - mask) * img_2

    return naive, new_img
