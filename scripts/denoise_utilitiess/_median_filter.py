import numpy as np


def calculate_median(array):
    """
    Return the median of a 1-D array.

    Parameters:
        array (np.ndarray): Input 1-D array.
    Returns:
        float: Median value.
    """
    sorted_array = np.sort(array) #timsort (O(nlogn))
    median = sorted_array[len(array)//2]
    return median


def level_B(z_min, z_med, z_max, z_xy, S_xy, S_max):
    """
    Level B: Check if current pixel is impulse noise.
    """
    if z_min < z_xy < z_max:
        return z_xy  # Current pixel is not impulse noise, keep original
    else:
        return z_med  # Current pixel is impulse noise, replace with median


def level_A(z_min, z_med, z_max, z_xy, S_xy, S_max):
    """
    Level A: Check if median is impulse noise and adapt window size.
    """
    if z_min < z_med < z_max:
        # Median is not impulse noise, go to Level B
        return level_B(z_min, z_med, z_max, z_xy, S_xy, S_max)
    else:
        # Median is impulse noise, increase window size
        S_xy += 2  # Increase the size of S_xy to the next odd value
        if S_xy <= S_max:  # Repeat process with larger window
            return level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            return z_med  # Maximum window size reached, return median


def get_window_stats(image, i, j, window_size):
    """
    Extract statistics from a window around pixel (i, j).

    Parameters:
        image (np.ndarray): Input 2D image.
        i (int): Row index.
        j (int): Column index.
        window_size (int): Size of the window.
    Returns:
        tuple: (z_min, z_med, z_max, z_xy)
    """
    pad_size = window_size // 2
    height, width = image.shape

    # Define window boundaries with padding
    row_start = max(0, i - pad_size)
    row_end = min(height, i + pad_size + 1)
    col_start = max(0, j - pad_size)
    col_end = min(width, j + pad_size + 1)

    # Extract window
    window = image[row_start:row_end, col_start:col_end]

    # Calculate statistics
    z_min = np.min(window)
    z_max = np.max(window)
    z_med = np.median(window)
    z_xy = image[i, j]  # Current pixel value

    return z_min, z_med, z_max, z_xy


def adaptive_median_filter(image, S_max=7):
    """
    Apply adaptive median filter to remove salt-and-pepper noise.

    Parameters:
        image (np.ndarray): Input image (2D).
        S_max (int): Maximum allowed window size (must be odd).
    Returns:
        np.ndarray: Filtered image.
    """
    if S_max % 2 == 0:
        S_max += 1  # Ensure S_max is odd

    # Create output image
    filtered_image = image.copy().astype(np.float64)
    height, width = image.shape

    # Process each pixel
    for i in range(height):
        for j in range(width):
            S_xy = 3  # Start with 3x3 window

            # Get initial window statistics
            z_min, z_med, z_max, z_xy = get_window_stats(image, i, j, S_xy)

            # Apply adaptive median filter algorithm
            filtered_value = level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
            filtered_image[i, j] = filtered_value

    return filtered_image.astype(image.dtype)
