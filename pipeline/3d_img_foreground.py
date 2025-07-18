"""
slice_denoise_flow.py
---------------------
This script provides functions to denoise a single slice from 3D CBCT data and generate a point cloud from the denoised image and mask. It is intended for medical image processing workflows, especially for extracting features from CBCT scans.
"""

import numpy as np
from scipy.signal import wiener
from numba import jit
from scripts.denoise_utilitiess import Denoise
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for environments without a display
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def three_d_denoising(cbct_path: str, mask_path: str, hu_min: int = -1000, hu_max: int = 3000, lower_threshold_mask: int = 170, low_threshold: int = 5, high_threshold: int = 15):
    """
    Denoise all slices from 3D CBCT data and their corresponding masks.

    Parameters:
        cbct_path (str): Path to the 3D CBCT .npy file.
        mask_path (str): Path to the 3D mask .npy file.
        hu_min (int, optional): Minimum Hounsfield value for scaling. Default is -1000.
        hu_max (int, optional): Maximum Hounsfield value for scaling. Default is 3000.

    Returns:
        denoised_3d (np.ndarray): Denoised and windowed 3D image (uint8).
        mask_3d (np.ndarray): Corresponding 3D mask.
    """
    logger.info(f"Loading CBCT data from {cbct_path}")
    cbct_data = np.load(cbct_path)
    logger.info(f"Loading mask data from {mask_path}")
    mask_data = np.load(mask_path)
    logger.info(f"CBCT data shape: {cbct_data.shape}")
    logger.info(f"Mask data shape: {mask_data.shape}")
    converted = Denoise.hounsfield_to_255(cbct_data, hu_min, hu_max)
    logger.info("Converted Hounsfield units to 0-255 scale.")
    num_slices = converted.shape[1]
    # denoised_slices = []
    edges_masked_slices = []
    masked_slices = []
    mask_3d_slices = []
    for slice_index in range(num_slices):
        logger.info(f"Processing slice {slice_index+1}/{num_slices}")
        slice_2d = converted[:, slice_index, :]
        mask_2d = mask_data[:, slice_index, :]
        output = Denoise.adaptive_median_filter(slice_2d, 7)
        logger.info(f"Applied adaptive median filter to slice {slice_index+1}")
        log_transformed, original_min, eps = Denoise.log_transform_for_denoising(output)
        logger.info(f"Applied log transform to slice {slice_index+1}")
        denoised_log = Denoise.denoise_log_transformed_image(log_transformed)
        logger.info(f"Denoised log-transformed image for slice {slice_index+1}")
        inv_log = Denoise.inverse_log_transform(denoised_log, original_min, eps)
        logger.info(f"Inverse log transform applied to slice {slice_index+1}")
        denoised_img_uint8 = Denoise.apply_window(inv_log, window_center=40, window_width=400)
        logger.info(f"Applied windowing to slice {slice_index+1}")
        # Thresholding
        _, mask = cv2.threshold(denoised_img_uint8, lower_threshold_mask, 255, cv2.THRESH_BINARY)
        logger.info(f"Applied thresholding to slice {slice_index+1}")
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        # Contour detection
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Detected {len(contours)} contours in slice {slice_index+1}")
        shape = mask_uint8.shape[:2]
        # Masking
        mask = Denoise.contour_to_filled_mask(contours, shape)
        logger.info(f"Created filled mask from contours for slice {slice_index+1}")
        mask_bool = (mask > 0).astype(np.uint8)
        masked_img = cv2.bitwise_and(denoised_img_uint8, denoised_img_uint8, mask=mask_bool)
        logger.info(f"Applied mask to denoised image for slice {slice_index+1}")
        # Edge detection
        edges_masked = Denoise.canny_edge_detection(masked_img, low_threshold=low_threshold, high_threshold=high_threshold)
        logger.info(f"Applied Canny edge detection to masked image for slice {slice_index+1}")
        # Save the 2D array as an image
        # plt.imsave(rf'D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\pc\slice_{slice_index+1}.png', edges_masked, cmap='gray')
        edges_masked_slices.append(edges_masked)
        # np.save(rf"D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\array\edges_masked_img_{slice_index+1}.npy", masked_img)
    denoised_3d = np.stack(edges_masked_slices, axis=1)
    mask_3d = mask_data
    logger.info("Stacked all denoised slices into a 3D volume.")
    # # Estimate noise variance from the 3D image (using median absolute deviation)
    # noise_sigma = np.median(np.abs(denoised_3d - np.median(denoised_3d))) / 0.6745
    # noise_var = noise_sigma ** 2
    # logger.info(f"Estimated noise variance: {noise_var:.6f}")
    # if noise_var < 1e-8:
    #     logger.warning("Estimated noise variance is very low; Wiener filter may have limited effect.")
    # denoised_3d = Denoise.three_d_adaptive_wiener_filter(denoised_3d, noise_var, window_size=5, axis=1)
    # logger.info("Applied 3D Wiener filtering to the denoised volume.")
    return denoised_3d, mask_3d

def pointcloud_formulation_3d(
    edge_detected_img, mask_3d, output_path_ply: str, output_path_npy: str
):
    """
    Generate a 3D point cloud from a 3D edge-detected image and mask, saving to .ply and .npy files.

    Parameters:
        edge_detected_img (np.ndarray): 3D edge-detected image (binary or intensity).
        mask_3d (np.ndarray): 3D mask corresponding to the image.
        output_path_ply (str): Path to save the output .ply file.
        output_path_npy (str): Path to save the output .npy file.

    Returns:
        None
    """
    # Compute 3D gradient magnitude
    dz, dy, dx = np.gradient(edge_detected_img.astype(float))
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    # Get coordinates of nonzero (edge) points
    edge_points = np.column_stack(np.where(edge_detected_img > 0))
    point_cloud = []

    for point in edge_points:
        z, y, x = point  # np.where order: (z, y, x)
        grad = magnitude[z, y, x]
        label = mask_3d[z, y, x]
        point_cloud.append((x, y, z, grad, int(label)))  # (x, y, z, gradient, label)

    num_points = len(point_cloud)
    print(f"Number of points detected: {num_points}")

    # Save to .ply file
    with open(output_path_ply, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float intensity_gradient\n")
        ply_file.write("property uchar label\n")
        ply_file.write("end_header\n")
        for point in point_cloud:
            x, y, z, grad, label = point
            ply_file.write(f"{x} {y} {z} {grad} {label}\n")

    print(f"Point cloud data saved to {output_path_ply}")

    # Save to .npy file
    point_cloud_array = np.array(point_cloud)
    np.save(output_path_npy, point_cloud_array)
    print(f"Point cloud saved as '{output_path_npy}'")

    return

if __name__ == "__main__":
    # denoised_img, mask_3d = three_d_denoising(
    #     cbct_path=r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\data.npy",
    #     mask_path=r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\gt_alpha.npy"
    # )
   
    # Save the selected slices (or the entire denoised_img if that's what you want)
    # np.save(r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy", denoised_img)

    denoised_img = np.load(r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy")
    mask_3d = np.load(r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\gt_alpha.npy")

    pointcloud_formulation_3d(denoised_img, mask_3d, r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_ply.ply", r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_npy.npy")
        