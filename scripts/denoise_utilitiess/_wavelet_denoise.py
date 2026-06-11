import pywt
from scipy.signal import wiener
import numpy as np


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
    noise_sigma = estimate_noise(details[-1][2])

    # 3. Apply Wiener filtering to low-frequency (approximation)
    cA_denoised = adaptive_wiener_filter(cA, noise_var=noise_sigma**2, window_size=window_size)

    # 4. Apply adaptive wavelet thresholding to high-frequency (details)
    details_denoised = []
    for (cH, cV, cD) in details:
        cH_denoised = adaptive_wavelet_threshold(cH, noise_sigma, threshold_factor)
        cV_denoised = adaptive_wavelet_threshold(cV, noise_sigma, threshold_factor)
        cD_denoised = adaptive_wavelet_threshold(cD, noise_sigma, threshold_factor)
        details_denoised.append((cH_denoised, cV_denoised, cD_denoised))

    # 5. Reconstruct the denoised image in the log domain
    coeffs_denoised = [cA_denoised] + details_denoised
    denoised_log_image = pywt.waverec2(coeffs_denoised, wavelet=wavelet)
    return denoised_log_image


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
