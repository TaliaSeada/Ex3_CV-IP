import cv2
import matplotlib.pyplot as plt
import numpy as np

from ex3_utils import *
import time
import warnings
warnings.filterwarnings('ignore')


def MSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(a - b).mean()

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("hierarchical Demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 3],
                  [0, 1, -5],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])

    st = time.time()
    uv, pts = opticalFlowPyrLK(img_1.astype(np.float), img_2.astype(np.float), 4, stepSize=20, winSize=5)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    displayOpticalFlowh(img_1, pts, uv)


def lkDemot(img_path, t):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemot(img_path, t):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    uv, pts = opticalFlowPyrLK(img_1.astype(np.float), img_2.astype(np.float), 4, stepSize=20, winSize=5)
    displayOpticalFlowh(img_1, pts, uv)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")
    t = np.array([[1, 0, 3],
                  [0, 1, -5],
                  [0, 0, 1]], dtype=np.float)
    lkDemot(img_path, t)
    hierarchicalkDemot(img_path, t)

    lk = 'output/lucas-kanade.png'
    lkh = 'output/hierarchical lucas-kanade.png'

    lk = cv2.cvtColor(cv2.imread(lk), cv2.COLOR_BGR2RGB)
    lk = cv2.resize(lk, (0, 0), fx=.5, fy=0.5)

    lkh = cv2.cvtColor(cv2.imread(lkh), cv2.COLOR_BGR2RGB)
    lkh = cv2.resize(lkh, (0, 0), fx=.5, fy=0.5)

    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].set_title("lucas-kanade")
    ax[1].set_title("hierarchical lucas-kanade")
    ax[0].imshow(lk)
    ax[1].imshow(lkh)
    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.savefig("lucas-kanade")
    plt.show()


def displayOpticalFlowh(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.savefig("hierarchical lucas-kanade")
    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def transLK(img):
    t_x = -.6
    t_y = -.5
    t = np.array([[1, 0, t_x],
                  [0, 1, t_y],
                  [0, 0, 1]], dtype=np.float)
    shifted1 = cv2.warpPerspective(img, t, img.shape[::-1])
    cv2.imwrite('input/imTransA2.jpg', shifted1)
    print("Translation LK")
    mat = findTranslationLK(img, shifted1)
    shifted2 = cv2.warpPerspective(img, mat, img.shape[::-1])
    print("MSE:", MSE(shifted1, shifted2))
    print(mat)

    plt.gray()
    f, ax = plt.subplots(1, 2)
    ax[0].set_title("translation with given")
    ax[1].set_title("translation with mine")
    ax[0].imshow(shifted1)
    ax[1].imshow(shifted2)
    plt.show()


def transCorr(img):
    t_x = -80
    t_y = 20
    t = np.array([[1, 0, t_x],
                  [0, 1, t_y],
                  [0, 0, 1]], dtype=np.float)
    shifted1 = cv2.warpPerspective(img, t, img.shape[::-1])
    cv2.imwrite("input/imTransB2.jpg", shifted1)
    print("Translation Correlation")
    mat = findTranslationCorr(img, shifted1)
    shifted2 = cv2.warpPerspective(img, mat, img.shape[::-1])
    print("MSE:", MSE(shifted1, shifted2))
    print(mat)

    plt.gray()
    f, ax = plt.subplots(1, 2)
    ax[0].set_title("translation with given")
    ax[1].set_title("translation with mine")
    ax[0].imshow(shifted1)
    ax[1].imshow(shifted2)
    plt.show()


def rigidLK(img):
    theta = .5
    t_x = -.2
    t_y = .6
    t = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
        [0, 0, 1]
    ])
    shifted1 = cv2.warpPerspective(img, t, img.shape[::-1])
    cv2.imwrite("input/imRigidA2.jpg", shifted1)
    print("Rigid Lk")
    mat = findRigidLK(img, shifted1)
    shifted2 = cv2.warpPerspective(img, mat, img.shape[::-1])
    print("MSE:", MSE(shifted1, shifted2))
    print(mat)

    plt.gray()
    f, ax = plt.subplots(1, 2)
    ax[0].set_title("rigid with given")
    ax[1].set_title("rigid with mine")
    ax[0].imshow(shifted1)
    ax[1].imshow(shifted2)
    plt.show()


def rigidCorr(img):
    theta = 10
    t_x = -6
    t_y = 11
    t = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
        [0, 0, 1]
    ])
    shifted1 = cv2.warpPerspective(img, t, img.shape[::-1])
    cv2.imwrite("output/imRigidB2.jpg", shifted1)
    print("Rigid Correlation")
    mat = findRigidCorr(img, shifted1)
    shifted2 = cv2.warpPerspective(img, mat, img.shape[::-1])
    print("MSE:", MSE(shifted1, shifted2))
    print(mat)

    plt.gray()
    f, ax = plt.subplots(1, 2)
    ax[0].set_title("rigid with given")
    ax[1].set_title("rigid with mine")
    ax[0].imshow(shifted1)
    ax[1].imshow(shifted2)
    plt.show()


def warpImage(img):
    print("Warp Images")
    T = np.array([[1, 0, 10.5],
                  [0, 1, -40.7],
                  [0, 0, 1]], dtype=np.float)

    im2 = cv2.warpPerspective(img, T, img.shape[::-1])
    im2_mine = warpImages(img, im2, T)

    print("MSE between my function and OpenCV function: ")
    print("MSE:", MSE(im2_mine, img))

    plt.gray()
    f, ax = plt.subplots(1, 2)

    ax[0].set_title("given image - translation")
    ax[1].set_title("my image - translation")
    ax[0].imshow(img)
    ax[1].imshow(im2_mine)
    plt.show()

    # second test
    theta = 30
    T = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 30],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), -40],
        [0, 0, 1]
    ])
    im2 = cv2.warpPerspective(img, T, img.shape[::-1])
    im2_mine = warpImages(img, im2, T)

    print("MSE between my function and OpenCV function: ")
    print("MSE:", MSE(im2_mine, img))

    plt.gray()
    f, ax = plt.subplots(1, 2)
    ax[0].set_title("given image - rigid")
    ax[1].set_title("my image - rigid")
    ax[0].imshow(img)
    ax[1].imshow(im2_mine)
    plt.show()


def imageWarpingDemo():
    """
    ADD TEST
    :return:
    """
    print("Image Warping Demo")
    img_path = 'input/imTransA1.jpg'
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=.5, fy=.5)

    img_path = 'input/imTransB1.jpg'
    img2 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (0, 0), fx=.5, fy=.5)

    img_path = 'input/imRigidA1.jpg'
    img3 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img3 = cv2.resize(img3, (0, 0), fx=.5, fy=.5)

    img_path = 'input/imRigidB1.jpg'
    img4 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img4 = cv2.resize(img4, (0, 0), fx=.5, fy=.5)

    transLK(img1)
    print("\n")
    transCorr(img2)
    print("\n")
    rigidLK(img3)
    print("\n")
    rigidCorr(img4)
    print("\n")
    warpImage(img2)


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('output/sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    print("\n")

    hierarchicalkDemo(img_path)
    print("\n")

    compareLK(img_path)
    print("\n")

    imageWarpingDemo()
    print("\n")

    pyrGaussianDemo('input/pyr_bit.jpg')
    print("\n")

    pyrLaplacianDemo('input/pyr_bit.jpg')
    print("\n")

    blendDemo()
    print("\n")


if __name__ == '__main__':
    main()
