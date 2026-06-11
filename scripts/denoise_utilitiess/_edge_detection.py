import cv2
import numpy as np
from scipy.ndimage import convolve


def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    """
    Apply Canny edge detection to an intensity image.

    Parameters:
        image (np.ndarray): Input image (uint8 scale).
        low_threshold (int): Lower bound for hysteresis thresholding.
        high_threshold (int): Upper bound for hysteresis thresholding.
    Returns:
        np.ndarray: Binary image with detected edges.
    """
    # Ensure image is uint8 for OpenCV
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, low_threshold, high_threshold)
    return edges


def prewitt_2d(image):
    """
    Apply the Prewitt edge detector to a 2D image.

    Parameters:
        image (np.ndarray): 2D input image (e.g., shape [H, W]).

    Returns:
        edges (np.ndarray): 2D array of edge intensity (same shape as input).
    """
    # Define Prewitt kernels for X and Y directions
    kernel_x = np.array([[ -1, 0, 1],
                        [ -1, 0, 1],
                        [ -1, 0, 1]])
    kernel_y = np.array([[ -1, -1, -1],
                        [  0,  0,  0],
                        [  1,  1,  1]])

    # Convolve with kernels
    gx = convolve(image.astype(float), kernel_x, mode='nearest')
    gy = convolve(image.astype(float), kernel_y, mode='nearest')

    # Edge magnitude
    edges = np.sqrt(gx**2 + gy**2)
    return edges


def prewitt_3d(volume):
    """
    Apply the Prewitt edge detector to a 3D volume.

    Parameters:
        volume (np.ndarray): 3D input volume (e.g., shape [D, H, W]).

    Returns:
        edges (np.ndarray): 3D array of edge intensity (same shape as input).
    """
    # Prewitt kernel for X gradient
    kernel_x = np.array([
        [[-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]],
        [[-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]],
        [[-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]]
    ])

    # Prewitt kernel for Y gradient
    kernel_y = np.array([
        [[-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]],
        [[-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]],
        [[-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]]
    ])

    # Prewitt kernel for Z gradient
    kernel_z = np.array([
        [[-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]
    ])

    # Calculate gradients along each axis
    gx = convolve(volume.astype(float), kernel_x, mode='nearest')
    gy = convolve(volume.astype(float), kernel_y, mode='nearest')
    gz = convolve(volume.astype(float), kernel_z, mode='nearest')

    # Edge magnitude
    edges = np.sqrt(gx**2 + gy**2 + gz**2)
    return edges
