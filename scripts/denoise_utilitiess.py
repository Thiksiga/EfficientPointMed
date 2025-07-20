"""
denoise_utilitiess.py
---------------------
This module provides denoising utilities for CBCT and medical image processing, including adaptive median filtering, log transformation, wavelet denoising, and non-local means filtering. It also provides edge detection and mask utilities for point cloud extraction.

Classes:
    Denoise: Static methods for denoising, filtering, and image processing.
    NLMeansCustom (template, commented): Example for custom non-local means filtering.
"""
import pywt
from scipy.signal import wiener
import numpy as np
from numba import jit
import cv2
import open3d as o3d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


class BasicPointCloud:
    def __init__(self, points, colors=None, normals=None):
        self.points = points
        self.colors = colors if colors is not None else np.zeros_like(points)
        self.normals = normals if normals is not None else np.zeros_like(points)

class Denoise:
    """
    Collection of static methods for denoising, filtering, and processing medical images.
    """
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def level_B(z_min, z_med, z_max, z_xy, S_xy, S_max):
        """
        Level B: Check if current pixel is impulse noise.
        """
        if z_min < z_xy < z_max:
            return z_xy  # Current pixel is not impulse noise, keep original
        else:
            return z_med  # Current pixel is impulse noise, replace with median

    @staticmethod
    def level_A(z_min, z_med, z_max, z_xy, S_xy, S_max):
        """
        Level A: Check if median is impulse noise and adapt window size.
        """
        if z_min < z_med < z_max:
            # Median is not impulse noise, go to Level B
            return Denoise.level_B(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            # Median is impulse noise, increase window size
            S_xy += 2  # Increase the size of S_xy to the next odd value
            if S_xy <= S_max:  # Repeat process with larger window
                return Denoise.level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
            else:
                return z_med  # Maximum window size reached, return median

    @staticmethod
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

    @staticmethod
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
                z_min, z_med, z_max, z_xy = Denoise.get_window_stats(image, i, j, S_xy)

                # Apply adaptive median filter algorithm
                filtered_value = Denoise.level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
                filtered_image[i, j] = filtered_value

        return filtered_image.astype(image.dtype)

    @staticmethod
    def log_transform_for_denoising(image):
        """
        Apply logarithmic transformation to convert multiplicative to additive noise.

        Parameters:
            image (np.ndarray): Input image.
        Returns:
            tuple: (log_image, original_min, epsilon)
        """
        epsilon = 1e-10
        original_min = np.min(image)
        if original_min <= 0:
            shifted_image = image - original_min + epsilon
        else:
            shifted_image = image + epsilon
            original_min = 0  # No shift needed

        log_image = np.log(shifted_image)
        return log_image, original_min, epsilon

    @staticmethod
    def inverse_log_transform(log_image, original_min, epsilon):
        """
        Convert back from log domain to original intensity domain.

        Parameters:
            log_image (np.ndarray): Log-transformed image.
            original_min (float): Minimum value used for shifting.
            epsilon (float): Small constant used for stability.
        Returns:
            np.ndarray: Reconstructed image.
        """
        reconstructed = np.exp(log_image)
        if original_min != 0:
            final_image = reconstructed + original_min - epsilon
        else:
            final_image = reconstructed - epsilon
        return final_image

    @staticmethod
    def adaptive_wiener_filter(approx, noise_var, window_size=5):
        """
        Apply Wiener filter to 2D input.

        Parameters:
            approx (np.ndarray): Input 2D array.
            noise_var (float): Estimated noise variance.
            window_size (int): Window size for filtering.
        Returns:
            np.ndarray: Wiener-filtered array.
        """
        # Wiener filter expects 2D input; noise_var is the estimated noise variance
        return wiener(approx, mysize=window_size, noise=noise_var)
    
    @staticmethod
    def adaptive_wavelet_threshold(detail, noise_sigma, threshold_factor=3):
        """
        Apply soft thresholding for adaptive wavelet thresholding.

        Parameters:
            detail (np.ndarray): Detail coefficients.
            noise_sigma (float): Estimated noise standard deviation.
            threshold_factor (float): Scaling factor for threshold.
        Returns:
            np.ndarray: Thresholded coefficients.
        """
        # Soft thresholding for adaptive wavelet thresholding
        T = threshold_factor * noise_sigma
        return np.sign(detail) * np.maximum(np.abs(detail) - T, 0)

    @staticmethod
    def estimate_noise(detail_coeffs):
        """
        Estimate noise using Median Absolute Deviation estimator.

        Parameters:
            detail_coeffs (np.ndarray): Detail coefficients.
        Returns:
            float: Estimated noise sigma.
        """
        # Median Absolute Deviation estimator for noise
        return np.median(np.abs(detail_coeffs)) / 0.6745

    @staticmethod
    def denoise_log_transformed_image(log_image, wavelet='db4', level=2, window_size=5, threshold_factor=3):
        """
        Denoise a log-transformed image using adaptive Wiener filtering and wavelet thresholding.

        Parameters:
            log_image (np.ndarray): Log-transformed image.
            wavelet (str): Wavelet type.
            level (int): Number of decomposition levels.
            window_size (int): Wiener filter window size.
            threshold_factor (float): Scaling factor for threshold.
        Returns:
            np.ndarray: Denoised image in log domain.
        """
        # 1. Wavelet decomposition
        coeffs = pywt.wavedec2(log_image, wavelet=wavelet, level=level)
        cA, details = coeffs[0], coeffs[1:]

        # 2. Estimate noise from the finest scale diagonal detail coefficients (cD)
        noise_sigma = Denoise.estimate_noise(details[-1][2])

        # 3. Apply Wiener filtering to low-frequency (approximation)
        cA_denoised = Denoise.adaptive_wiener_filter(cA, noise_var=noise_sigma**2, window_size=window_size)

        # 4. Apply adaptive wavelet thresholding to high-frequency (details)
        details_denoised = []
        for (cH, cV, cD) in details:
            cH_denoised = Denoise.adaptive_wavelet_threshold(cH, noise_sigma, threshold_factor)
            cV_denoised = Denoise.adaptive_wavelet_threshold(cV, noise_sigma, threshold_factor)
            cD_denoised = Denoise.adaptive_wavelet_threshold(cD, noise_sigma, threshold_factor)
            details_denoised.append((cH_denoised, cV_denoised, cD_denoised))

        # 5. Reconstruct the denoised image in the log domain
        coeffs_denoised = [cA_denoised] + details_denoised
        denoised_log_image = pywt.waverec2(coeffs_denoised, wavelet=wavelet)
        return denoised_log_image

    @staticmethod
    def three_d_adaptive_wiener_filter(volume, noise_var, window_size=5, axis=1):
        """
        Apply Wiener filter to a 3D volume slice-by-slice along the specified axis.

        Parameters:
            volume (np.ndarray): Input 3D array.
            noise_var (float): Estimated noise variance.
            window_size (int): Window size for filtering.
            axis (int): Axis along which to apply the filter (default 1).
        Returns:
            np.ndarray: Wiener-filtered 3D array.
        """
        from scipy.signal import wiener
        epsilon = 1e-8
        safe_noise_var = noise_var + epsilon
        filtered = np.empty_like(volume, dtype=np.float64)
        # Move the axis to the middle for easy iteration
        volume = np.moveaxis(volume, axis, 0)
        filtered = np.empty_like(volume, dtype=np.float64)
        for i in range(volume.shape[0]):
            filtered[i] = wiener(volume[i], mysize=window_size, noise=safe_noise_var)
        # Move axis back to original
        filtered = np.moveaxis(filtered, 0, axis)
        return filtered.astype(volume.dtype)

    # Function to preprocess neighbors (small_window x small_window) for each pixel
    @staticmethod
    @jit(nopython=True)
    def findAllNeighbors(padImg, small_window, big_window, h, w):
        """
        Preprocess neighbors (small_window x small_window) for each pixel.
        """
        smallWidth = small_window // 2
        bigWidth = big_window // 2
        neighbors = np.zeros((padImg.shape[0], padImg.shape[1], small_window, small_window))
        for i in range(bigWidth, bigWidth + h):
            for j in range(bigWidth, bigWidth + w):
                neighbors[i, j] = padImg[(i - smallWidth):(i + smallWidth + 1), (j - smallWidth):(j + smallWidth + 1)]
        return neighbors

    # Function to calculate the weighted average value (Ip) for each pixel
    @staticmethod
    @jit(nopython=True)
    def evaluateNorm(pixelWindow, neighborWindow, Nw):
        """
        Calculate the weighted average value (Ip) for each pixel.
        """
        Ip_Numerator, Z = 0.0, 0.0
        for i in range(neighborWindow.shape[0]):
            for j in range(neighborWindow.shape[1]):
                q_window = neighborWindow[i, j]
                q_x, q_y = q_window.shape[0] // 2, q_window.shape[1] // 2
                Iq = q_window[q_x, q_y]
                w = np.exp(-1 * ((np.sum((pixelWindow - q_window) ** 2)) / Nw))
                Ip_Numerator += w * Iq
                Z += w
        return Ip_Numerator / Z if Z > 0 else 0.0

    @staticmethod
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

    @staticmethod
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
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def curvature_preserving_sampling(point_cloud: BasicPointCloud, k_neighbors=20, fraction=0.2):
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
        o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
        curvatures = []
        kdtree = o3d.geometry.KDTreeFlann(o3d_cloud)
        for i in range(len(point_cloud.points)):
            _, idx, _ = kdtree.search_knn_vector_3d(o3d_cloud.points[i], k_neighbors)
            neighbors = np.asarray(o3d_cloud.points)[idx]
            cov = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(cov)
            curvature = eigenvalues[0] / sum(eigenvalues)
            curvatures.append(curvature)
        sorted_indices = np.argsort(curvatures)[::-1]
        num_samples = int(len(point_cloud.points) * fraction)
        selected_indices = sorted_indices[:num_samples]
        return BasicPointCloud(points=point_cloud.points[selected_indices],
                            colors=point_cloud.colors[selected_indices],
                            normals=point_cloud.normals[selected_indices]), selected_indices

  





    




# Template for a custom Non-Local Means implementation (currently commented out)
# class NLMeansCustom:
#     """
#     Custom Non-Local Means implementation for grayscale images.
#     """
#     def apply(self, img, h=30, small_window=7, big_window=21):
#         """
#         Apply NLM filtering to a 2D grayscale image.
#         Parameters:
#             img (np.ndarray): Grayscale image.
#             h (float): Filtering parameter (higher removes more noise).
#             small_window (int): Patch size.
#             big_window (int): Search window size.
#         Returns:
#             np.ndarray: Denoised image (float64).
#         """
#         # Padding the original image with reflect mode
#         padImg = np.pad(img, big_window // 2, mode='reflect')
#         return self.NLM(padImg, img, h, small_window, big_window)
#
#     @staticmethod
#     def NLM(padImg, img, h, small_window, big_window):
#         """
#         Core NLM algorithm for denoising.
#         """
#         Nw = (h ** 2) * (small_window ** 2)
#         h_img, w_img = img.shape
#         result = np.zeros(img.shape)
#         bigWidth = big_window // 2
#         smallWidth = small_window // 2
#         neighbors = Denoise.findAllNeighbors(padImg, small_window, big_window, h_img, w_img)
#         for i in range(bigWidth, bigWidth + h_img):
#             for j in range(bigWidth, bigWidth + w_img):
#                 pixelWindow = neighbors[i, j]
#                 neighborWindow = neighbors[(i - bigWidth):(i + bigWidth + 1), (j - bigWidth):(j + bigWidth + 1)]
#                 Ip = Denoise.evaluateNorm(pixelWindow, neighborWindow, Nw)
#                 result[i - bigWidth, j - bigWidth] = max(min(255, Ip), 0)
#         return result




