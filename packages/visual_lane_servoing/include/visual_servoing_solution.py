from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_left = np.zeros(shape)

    half = shape[1] // 2
    # Sol1
    # steer_matrix_left[:, :half] = -0.3
    # steer_matrix_left[: half // 4 + 1, :half] = 0.1

    # Sol2
    steer_matrix_left[int(shape[0] * 5 / 8) :, :half] = -0.01

    # Sol3
    steer_matrix_left[:, shape[1] // 3 :] = -1

    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_right = np.zeros(shape)

    # Sol1
    half = shape[1] // 2
    # steer_matrix_right[half // 4 + 1 :, half:] = 0.2
    # steer_matrix_right[: half // 4 + 1, half:] = 0.1

    # Sol2
    # steer_matrix_right[int(shape[0] * 5 / 8) :, half:] = 0.01

    # Sol3
    steer_matrix_right[:, : shape[1] * 2 // 3] = 1

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    # TODO: implement your own solution here
    height, width, _ = image.shape

    # Parameters to play around with
    sigma_gaussian_blur = 4
    white_lower_hsv = np.array([0, 0, 120])
    white_upper_hsv = np.array([180, 75, 255])
    yellow_lower_hsv = np.array([15, 60, 60])
    yellow_upper_hsv = np.array([45, 255, 255])

    # 0) Processing the image
    imgbgr = image
    # OpenCV uses BGR by default, whereas matplotlib uses RGB
    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)
    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

    # 1) (Optional) Find the horizon and remove it
    # mask_ground = np.ones((height, width), dtype=np.uint8)
    # mask_ground[:175, :] = 0

    # 2) Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img, (0, 0), sigma_gaussian_blur)

    # 3) Convolve the image with the Sobel operator (filter) to compute the
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)
    Gmag = np.sqrt(sobelx * sobelx + sobely * sobely)  # magnitude of gradients

    # 4) Make a mask to clip filter out weaker gradients (form of non-maximal suppression)
    threshold = np.max(Gmag) // 5
    mask_mag = Gmag > threshold

    # 5) Let's create masks for the left and right halves of the image
    mask_left = np.ones(sobelx.shape)
    mask_left[:, int(np.floor(width / 2)) : width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:, 0 : int(np.floor(width / 2))] = 0

    # 6) Generate a mask that identifies pixels based on the sign of their x-derivative
    mask_sobelx_pos = sobelx > 0
    mask_sobelx_neg = sobelx < 0
    mask_sobely_pos = sobely > 0
    mask_sobely_neg = sobely < 0

    # 7) White and yellow masks
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)

    # Let's generate the complete set of masks, including those based on color
    mask_left_edge = (
        mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    )
    mask_right_edge = (
        mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
    )

    return mask_left_edge, mask_right_edge
