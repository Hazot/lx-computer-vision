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
    height, width = shape
    left_region_width = width // 5  # Adjustable width
    decay_rate = 0.002
    weight = -1.6

    # Generate the decay values along the vertical axis
    decay_values = weight + np.exp(decay_rate * np.arange(height // 2, height))
    decay_values *= -1

    # Apply decay values to the left region in the bottom half of the matrix
    steer_matrix_left[height // 2 :, width // 2 - left_region_width : width // 2] = (
        decay_values[:, np.newaxis]
    )

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
    height, width = shape
    right_region_width = width // 5  # Adjustable widthf
    decay_rate = 0.002
    weight = -1.6

    # Generate the decay values along the vertical axis
    decay_values = (weight + np.exp(decay_rate * np.arange(height // 2, height)))

    # Apply decay values to the right region in the bottom half of the matrix
    steer_matrix_right[height // 2 :, width // 2 : width // 2 + right_region_width] = (
        decay_values[:, np.newaxis]
    )

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
    sigma = 4

    # Colors
    white_lower_hsv = np.array([0, 0, 167])
    white_upper_hsv = np.array([179, 65, 255])
    yellow_lower_hsv = np.array([20, 60, 75])
    yellow_upper_hsv = np.array([45, 255, 255])

    # 1) Processing the image
    imgbgr = image
    imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

    # 1.5) (Optional) Find the horizon and remove it
    H = np.array(
        [
            -4.137917960301845e-05,
            -0.00011445854191468058,
            -0.1595567007347241,
            0.0008382870319844166,
            -4.141689222457687e-05,
            -0.2518201638170328,
            -0.00023561657746150284,
            -0.005370140574116084,
            0.9999999999999999,
        ]
    )

    H = np.reshape(H, (3, 3))
    Hinv = np.linalg.inv(H)

    mask_ground = np.ones(img.shape, dtype=np.uint8)
    mask_ground = cv2.warpPerspective(mask_ground, Hinv, (width, height))
    mask_ground[:175] = 0

    # 2) Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img, (0, 0), sigma)

    # 3) Convolve the image with the Sobel operator (filter) to compute the
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)
    Gmag = np.sqrt(sobelx * sobelx + sobely * sobely)  # magnitude of gradients

    # 4) Make a mask to clip filter out weaker gradients (form of non-maximal suppression)
    # percentile_threshold_yellow = 90  # To keep the top 10% (if set to 90)
    # threshold_white = np.percentile(Gmag, percentile_threshold_yellow)

    # percentile_threshold_yellow = 85
    # threshold_yellow = np.percentile(Gmag, percentile_threshold_yellow)
    threshold_yellow = np.max(Gmag) // 6
    threshold_white = np.max(Gmag) // 4
    mask_mag_yellow = Gmag > threshold_yellow
    mask_mag_white = Gmag > threshold_white

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
        mask_ground
        * mask_left
        * mask_mag_yellow
        * mask_sobelx_neg
        * mask_sobely_neg
        * mask_yellow
    )
    mask_right_edge = (
        mask_ground
        * mask_right
        * mask_mag_white
        * mask_sobelx_pos
        * mask_sobely_neg
        * mask_white
    )

    return mask_left_edge, mask_right_edge
