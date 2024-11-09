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

    width = shape[0]
    half = shape[1] // 2
    # Sol1
    # steer_matrix_left[:, :half] = -0.3
    # steer_matrix_left[: half // 4 + 1, :half] = 0.1

    # Create a gradient that starts high from the middle bottom and decreases toward the left top corner
    for y in range(width):
        for x in range(half):
            # Calculate weight based on distance from middle bottom
            weight = 0.3 * ((width - y) / width) * ((half - x) / half)
            steer_matrix_left[y, x] = weight
    
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
    width = shape[0]
    # steer_matrix_right[half // 4 + 1 :, half:] = 0.2
    # steer_matrix_right[: half // 4 + 1, half:] = 0.1

    # Create a gradient that starts high from the middle bottom and decreases toward the right top corner
    for y in range(width):
        for x in range(half, shape[1]):
            # Calculate weight based on distance from middle bottom
            weight = -0.2 * ((width - y) / width) * ((x - half) / half)
            steer_matrix_right[y, x] = weight
    
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
    percentile_threshold = 90  # To keep the top 10% (if set to 90)
    white_sensitivity = 75
    yellow_sensitivity = 60

    # Parameters (hard) for colors
    white_lower_hsv = np.array([0, 0, 255 - white_sensitivity])
    white_upper_hsv = np.array([179, white_sensitivity, 255])
    yellow_lower_hsv = np.array([15, yellow_sensitivity, yellow_sensitivity])
    yellow_upper_hsv = np.array([45, 255, 255])

    # 1) Processing the image
    imgbgr = image
    imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

    # 1.5) (Optional) Find the horizon and remove it
    # Naive way
    mask_ground = np.ones((height, width), dtype=np.uint8)
    mask_ground[:200, :] = 0

    # 2) Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img, (0, 0), sigma)

    # 3) Convolve the image with the Sobel operator (filter) to compute the
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)
    Gmag = np.sqrt(sobelx * sobelx + sobely * sobely)  # magnitude of gradients

    # 4) Make a mask to clip filter out weaker gradients (form of non-maximal suppression)
    # threshold = np.percentile(Gmag, percentile_threshold)
    threshold = np.max(Gmag) // 5  # Testing with this threshold to use most gradient
    # threshold = 40
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
        mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    )
    mask_right_edge = (
        mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
    )

    return mask_left_edge, mask_right_edge
