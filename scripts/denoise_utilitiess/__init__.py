"""
denoise_utilitiess.py
---------------------
This module provides denoising utilities for CBCT and medical image processing, including adaptive median filtering,
log transformation, wavelet denoising, and non-local means filtering. It also provides edge detection and mask utilities
for point cloud extraction.
 
Classes:
    Denoise: Static methods for denoising, filtering, and image processing.
    NLMeansCustom (template, commented): Example for custom non-local means filtering.
"""

from ._basic_pointcloud import BasicPointCloud
from ._intensity import hounsfield_to_255, apply_window
from ._median_filter import calculate_median, level_A, level_B, get_window_stats, adaptive_median_filter
from ._wavelet_denoise import (
    log_transform_for_denoising, inverse_log_transform,
    adaptive_wiener_filter, adaptive_wavelet_threshold,
    estimate_noise, denoise_log_transformed_image,
    three_d_adaptive_wiener_filter
)
from ._nlm_helpers import findAllNeighbors, evaluateNorm
from ._edge_detection import canny_edge_detection, prewitt_2d, prewitt_3d
from ._mask_utils import contour_to_filled_mask
from ._pointcloud import curvature_preserving_sampling, curvature_preserving_sampling_2


class Denoise:
    """
    Collection of static methods for denoising, filtering, and processing medical images.
    """
    hounsfield_to_255 = staticmethod(hounsfield_to_255)
    calculate_median = staticmethod(calculate_median)
    level_B = staticmethod(level_B)
    level_A = staticmethod(level_A)
    get_window_stats = staticmethod(get_window_stats)
    adaptive_median_filter = staticmethod(adaptive_median_filter)
    log_transform_for_denoising = staticmethod(log_transform_for_denoising)
    inverse_log_transform = staticmethod(inverse_log_transform)
    adaptive_wiener_filter = staticmethod(adaptive_wiener_filter)
    adaptive_wavelet_threshold = staticmethod(adaptive_wavelet_threshold)
    estimate_noise = staticmethod(estimate_noise)
    denoise_log_transformed_image = staticmethod(denoise_log_transformed_image)
    three_d_adaptive_wiener_filter = staticmethod(three_d_adaptive_wiener_filter)
    findAllNeighbors = staticmethod(findAllNeighbors)
    evaluateNorm = staticmethod(evaluateNorm)
    canny_edge_detection = staticmethod(canny_edge_detection)
    apply_window = staticmethod(apply_window)
    contour_to_filled_mask = staticmethod(contour_to_filled_mask)
    prewitt_2d = staticmethod(prewitt_2d)
    prewitt_3d = staticmethod(prewitt_3d)
    curvature_preserving_sampling = staticmethod(curvature_preserving_sampling)
    curvature_preserving_sampling_2 = staticmethod(curvature_preserving_sampling_2)
