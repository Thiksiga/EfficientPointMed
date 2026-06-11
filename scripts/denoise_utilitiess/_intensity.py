import numpy as np


def hounsfield_to_255(hounsfield_array, hu_min=-1000, hu_max=3000):
    """
    Convert Hounsfield units to 0-255 intensity values.

    Parameters:
        hounsfield_array (np.ndarray): Array with Hounsfield values.
        hu_min (int): Minimum Hounsfield value to map to 0.
        hu_max (int): Maximum Hounsfield value to map to 255.

    Returns:
        np.ndarray: Values scaled to 0-255 (uint8).
    """
    # Clip values to the specified range
    clipped = np.clip(hounsfield_array, hu_min, hu_max)

    # Scale to 0-255 range
    scaled = ((clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

    return scaled


def apply_window(img, window_center, window_width):
    """
    Apply windowing to an image for intensity normalization.

    Parameters:
        img (np.ndarray): Input image.
        window_center (int): Center of the window.
        window_width (int): Width of the window.
    Returns:
        np.ndarray: Windowed and normalized image (uint8).
    """
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    img_windowed = np.clip(img, min_val, max_val)
    img_norm = ((img_windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return img_norm
