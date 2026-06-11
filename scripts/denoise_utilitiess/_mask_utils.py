import cv2
import numpy as np


def contour_to_filled_mask(contour, shape, fill_value=1):
    """
    Fill the given contour and return the filled mask.

    Parameters:
        contour (np.ndarray): Contour as returned by cv2.findContours.
        shape (tuple): (height, width) specifying the mask size.
        fill_value (int): Value to fill inside contour (default 1).
    Returns:
        np.ndarray: Mask with filled contour (fill_value inside, 0 outside).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contour, -1, fill_value, thickness=-1)
    return mask
