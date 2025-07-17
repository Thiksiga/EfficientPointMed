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

def three_d_denoising(cbct_path: str, mask_path: str, hu_min: int = -1000, hu_max: int = 3000):
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
    cbct_data = np.load(cbct_path)
    mask_data = np.load(mask_path)
    print(f"CBCT data shape: {cbct_data.shape}")
    print(f"Mask data shape: {mask_data.shape}")
    converted = Denoise.hounsfield_to_255(cbct_data, hu_min, hu_max)
    num_slices = converted.shape[1]
    denoised_slices = []
    mask_slices = []
    for slice_index in range(num_slices):
        print(slice_index)
        slice_2d = converted[:, slice_index, :]
        mask_2d = mask_data[:, slice_index, :]
        output = Denoise.adaptive_median_filter(slice_2d, 7)
        print("adaptive_median_filter")
        log_transformed, original_min, eps = Denoise.log_transform_for_denoising(output)
        print("log_transform_for_denoising")
        denoised_log = Denoise.denoise_log_transformed_image(log_transformed)
        print("denoise_log_transformed_image")
        inv_log = Denoise.inverse_log_transform(denoised_log, original_min, eps)
        print("inverse_log_transform")
        denoised_img_uint8 = Denoise.apply_window(inv_log, window_center=40, window_width=400)
        denoised_slices.append(denoised_img_uint8)
        mask_slices.append(mask_2d)
        # np.save(rf"D:\Mphil\Programming\EfficientPointMed\output\3d\test\selected_denoised_3d{slice_index}.npy", denoised_img_uint8)
    denoised_3d = np.stack(denoised_slices, axis=1)
    mask_3d = np.stack(mask_slices, axis=1)
    # Estimate noise variance from the 3D image (using median absolute deviation)
    noise_sigma = np.median(np.abs(denoised_3d - np.median(denoised_3d))) / 0.6745
    noise_var = noise_sigma ** 2
    denoised_3d = Denoise.three_d_adaptive_wiener_filter(denoised_3d, noise_var, window_size=5, axis=1)
    return denoised_3d, mask_3d

def pointcloud_formulation(
    denoised_img, mask_2d, output_path_ply: str, output_path_npy: str,
    lower_threshold_mask: int = 170, low_threshold: int = 5, high_threshold: int = 15
):
    """
    Generate a point cloud from a denoised image and mask, saving to .ply and .npy files.

    Parameters:
        denoised_img (np.ndarray): Denoised 2D image (uint8).
        mask_2d (np.ndarray): 2D mask corresponding to the image.
        output_path_ply (str): Path to save the output .ply file.
        output_path_npy (str): Path to save the output .npy file.
        lower_threshold_mask (int, optional): Threshold for mask binarization. Default is 170.
        low_threshold (int, optional): Low threshold for Canny edge detection. Default is 5.
        high_threshold (int, optional): High threshold for Canny edge detection. Default is 15.

    Returns:
        None
    """
    _, mask = cv2.threshold(denoised_img, lower_threshold_mask, 255, cv2.THRESH_BINARY)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape = mask_uint8.shape[:2]
    mask = Denoise.contour_to_filled_mask(contours, shape)
    mask_bool = (mask > 0).astype(np.uint8)
    masked_img = cv2.bitwise_and(denoised_img, denoised_img, mask=mask_bool)
    edges_masked = Denoise.canny_edge_detection(masked_img, low_threshold=low_threshold, high_threshold=high_threshold)
    gradient = np.gradient(masked_img)
    magnitude = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    edge_points = np.column_stack(np.where(edges_masked > 0))
    point_cloud = []
    for point in edge_points:
        x, y = point
        grad = magnitude[x, y]
        label = mask_2d[x, y]
        point_cloud.append((y, x, grad, label))
    num_points = len(edge_points)
    print(f"Number of points detected: {num_points}")
    with open(output_path_ply, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(point_cloud)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float intensity_gradient\n")
        ply_file.write("property uchar label\n")
        ply_file.write("end_header\n")
        for point in point_cloud:
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")
    print("Point cloud data saved to point_cloud_data_with_labels.ply")
    point_cloud_array = np.array(point_cloud)
    np.save(output_path_npy, point_cloud_array)
    print("Point cloud saved as 'point_cloud.npy'")
    return

if __name__ == "__main__":
    denoised_img, mask_2d = three_d_denoising(
        cbct_path=r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\data.npy",
        mask_path=r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\gt_alpha.npy"
    )
   
    # Save the selected slices (or the entire denoised_img if that's what you want)
    np.save(r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy", denoised_img)
        