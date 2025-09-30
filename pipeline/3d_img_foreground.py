"""
slice_denoise_flow.py
---------------------
This script provides functions to denoise a single slice from 3D CBCT data and generate a point cloud from the denoised image and mask. It is intended for medical image processing workflows, especially for extracting features from CBCT scans.
"""

import numpy as np
from scipy.signal import wiener
from numba import jit
from scripts.denoise_utilitiess import Denoise, BasicPointCloud
import cv2
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for environments without a display
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import cv2

# def three_d_denoising(cbct_path: str, mask_path: str, pointcloud_save_path: str, hu_min: int = -1000, hu_max: int = 3000,
#                       lower_threshold_mask: int = 170, low_threshold: int = 0, high_threshold: int = 5):
#     """
#     Denoise all slices from 3D CBCT data and their corresponding masks, build a 3D point cloud.

#     Parameters:
#         cbct_path (str): Path to the 3D CBCT .npy file.
#         mask_path (str): Path to the 3D mask .npy file.
#         hu_min (int, optional): Minimum Hounsfield value for scaling. Default is -1000.
#         hu_max (int, optional): Maximum Hounsfield value for scaling. Default is 3000.
#         pointcloud_save_path (str, optional): If given, save full 3D point cloud as .npy at this path.

#     Returns:
#         denoised_3d (np.ndarray): Denoised and windowed 3D image (uint8).
#         mask_3d (np.ndarray): Corresponding 3D mask.
#         point_cloud_3d (np.ndarray): (N, 5) array: x, y, z, grad, label.
#     """
#     logger.info(f"Loading CBCT data from {cbct_path}")
#     cbct_data = np.load(cbct_path)
#     logger.info(f"Loading mask data from {mask_path}")
#     mask_data = np.load(mask_path)
#     logger.info(f"CBCT data shape: {cbct_data.shape}")
#     logger.info(f"Mask data shape: {mask_data.shape}")
#     converted = Denoise.hounsfield_to_255(cbct_data, hu_min, hu_max)
#     logger.info("Converted Hounsfield units to 0-255 scale.")
#     num_slices = converted.shape[1]

#     masked_slices = []
#     point_cloud_3d = []
#     sampled_pc_3d = []
#     sampled_pc_mask_3d = []

#     for slice_index in range(num_slices):
#         logger.info(f"Processing slice {slice_index+1}/{num_slices}")
#         slice_2d = converted[:, slice_index, :]
#         mask_2d = mask_data[:, slice_index, :]
#         output = Denoise.adaptive_median_filter(slice_2d, 7)
#         logger.info(f"Applied adaptive median filter to slice {slice_index+1}")
#         log_transformed, original_min, eps = Denoise.log_transform_for_denoising(output)
#         logger.info(f"Applied log transform to slice {slice_index+1}")
#         denoised_log = Denoise.denoise_log_transformed_image(log_transformed)
#         logger.info(f"Denoised log-transformed image for slice {slice_index+1}")
#         inv_log = Denoise.inverse_log_transform(denoised_log, original_min, eps)
#         logger.info(f"Inverse log transform applied to slice {slice_index+1}")
#         denoised_img_uint8 = Denoise.apply_window(inv_log, window_center=40, window_width=400)
#         logger.info(f"Applied windowing to slice {slice_index+1}")
#         # Thresholding
#         _, mask = cv2.threshold(denoised_img_uint8, lower_threshold_mask, 255, cv2.THRESH_BINARY)
#         logger.info(f"Applied thresholding to slice {slice_index+1}")
#         mask_uint8 = (mask > 0).astype(np.uint8) * 255
#         # Contour detection
#         contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         two_largest = sorted_contours[:2]
#         selected_contours = []
#         areas = []
#         if two_largest:
#             area1 = cv2.contourArea(two_largest[0])
#             areas.append(area1)
#             if area1 > 11000:
#                 selected_contours = [two_largest[0]]
#             else:
#                 if len(two_largest) > 1:
#                     area2 = cv2.contourArea(two_largest[1])
#                     areas.append(area2)
#                     selected_contours = two_largest
#                 else:
#                     selected_contours = [two_largest[0]]
#         logger.info(f"Detected {len(selected_contours)} contours in slice {slice_index+1}")
#         shape = mask_uint8.shape[:2]
#         mask = Denoise.contour_to_filled_mask(selected_contours, shape)
#         slice_2d_uint8 = slice_2d.astype(np.uint8)
#         mask_uint8 = (mask > 0).astype(np.uint8)
#         mask_cv = mask_uint8 * 255
#         masked_img = cv2.bitwise_and(slice_2d_uint8, slice_2d_uint8, mask=mask_cv)
#         logger.info(f"Applied mask to denoised image for slice {slice_index+1}")
#         edges_2d = Denoise.prewitt_2d(masked_img)
#         threshold = 5
#         edge_mask = edges_2d > threshold
#         edge_points = np.argwhere(edge_mask)  # Each row: [x, y]
#         # 3D Point Cloud: (x, y, z, grad, label)
#         for x, y in edge_points:
#             grad = edges_2d[x, y]
#             label = mask_2d[x, y]
#             # point_cloud_3d.append((x, y, slice_index, grad, label))  # x, y, z, grad, label
#         num_points = len(edge_points)
#         logger.info(f"Number of points detected in slice {slice_index+1}: {num_points}")
#         # Your point cloud: list of (y, x, grad, label)
#         points = np.array([(x, y, z, grad) for (x, y, z, grad, label) in point_cloud_3d])
#         labels = np.array([label for (x, y, z, grad, label) in point_cloud_3d])

#         pc = BasicPointCloud(points=points)

#         sampled_pc, selected_indices = Denoise.curvature_preserving_sampling(pc, k_neighbors=5, fraction=0.05)
#         sampled_labels = labels[selected_indices]
#         logger.info("Number of sampled points:", len(sampled_pc.points))

#         sampled_pc_3d.append(sampled_pc)
#         sampled_pc_mask_3d.append(sampled_labels)

#         # masked_slices.append(masked_img)
#         np.save(rf"D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\array\masked_img_{slice_index+1}.npy", sampled_pc_3d)

#     # denoised_3d = np.stack(masked_slices, axis=1)
#     # mask_3d = mask_data
#     # logger.info("Stacked all denoised slices into a 3D volume.")


#     point_cloud_3d = np.array(point_cloud_3d, dtype=np.float32)  # (N, 5)
#     if pointcloud_save_path is not None:
#         np.save(pointcloud_save_path, point_cloud_3d)
#         logger.info(f"Saved full 3D point cloud to {pointcloud_save_path}")

#     return denoised_3d, mask_3d, point_cloud_3d




# def pointcloud_formulation_3d(
#     edge_detected_img, mask_3d, output_path_sampled_npy: str, output_path_npy: str
# ):
#     """
#     Generate a 3D point cloud from a 3D edge-detected image and mask, saving to .npy files.

#     Parameters:
#         edge_detected_img (np.ndarray): 3D edge-detected image (binary or intensity).
#         mask_3d (np.ndarray): 3D mask corresponding to the image.
#         output_path_sampled_npy (str): Path to save the sampled point cloud (.npy).
#         output_path_npy (str): Path to save the full point cloud (.npy).

#     Returns:
#         None
#     """
#     # Compute edge gradients (use your Denoise.prewitt_3d or similar function)
#     edges_3d = Denoise.prewitt_3d(edge_detected_img)

#     # Threshold to get edge voxels
#     threshold = 5  # You may want to expose this as a parameter
#     edge_mask = edges_3d > threshold

#     # Get coordinates of edge voxels
#     edge_voxels = np.argwhere(edge_mask)  # Each row: [z, y, x]

#     # Build full point cloud: (x, y, z, grad, label)
#     point_cloud = []
#     for z, y, x in edge_voxels:
#         grad = edges_3d[z, y, x]
#         label = mask_3d[z, y, x]
#         point_cloud.append((x, y, z, grad, int(label)))

#     if not point_cloud:
#         print("No edge points found above threshold.")
#         return

#     # Convert to numpy arrays for sampling
#     points_xyz = np.array([[x, y, z] for (x, y, z, grad, label) in point_cloud], dtype=np.float32)
#     grads = np.array([grad for (x, y, z, grad, label) in point_cloud], dtype=np.float32)
#     labels = np.array([label for (x, y, z, grad, label) in point_cloud], dtype=np.uint8)

#     # Sample using curvature-preserving sampling (from your Denoise/BasicPointCloud classes)
#     pc = BasicPointCloud(points=points_xyz)
#     sampled_pc, selected_indices = Denoise.curvature_preserving_sampling(pc, k_neighbors=5, fraction=0.05)
#     sampled_labels = labels[selected_indices]
#     sampled_grads = grads[selected_indices]
#     sampled_xyz = sampled_pc.points

#     # Save sampled point cloud as .npy (shape N x 5: x, y, z, grad, label)
#     sampled_array = np.column_stack((sampled_xyz, sampled_grads, sampled_labels))
#     np.save(output_path_sampled_npy, sampled_array)
#     print(f"Sampled point cloud saved to {output_path_sampled_npy}")

#     # Save full point cloud (x, y, z, grad, label) as .npy
#     point_cloud_array = np.array(point_cloud, dtype=np.float32)
#     np.save(output_path_npy, point_cloud_array)
#     print(f"Full point cloud saved as '{output_path_npy}'")

#     print(f"Number of sampled points: {len(sampled_array)}")
#     print(f"Number of points detected: {len(point_cloud)}")

#     return
###--------Curvature preserving on 3d pointcloud -------------
import numpy as np
import cv2

def three_d_denoising(cbct_path: str, mask_path: str, pointcloud_save_path: str,
                      hu_min: int = -1000, hu_max: int = 3000,
                      lower_threshold_mask: int = 170, low_threshold: int = 0, high_threshold: int = 5,
                      sampling_fraction: float = 0.2, k_neighbors: int = 5):
    """
    Denoise all slices from 3D CBCT data and their corresponding masks, build a sampled 3D point cloud (slice-wise sampling).

    Parameters:
        cbct_path (str): Path to the 3D CBCT .npy file.
        mask_path (str): Path to the 3D mask .npy file.
        pointcloud_save_path (str): Path to save the full sampled 3D point cloud (.npy).
        hu_min (int, optional): Minimum Hounsfield value for scaling. Default is -1000.
        hu_max (int, optional): Maximum Hounsfield value for scaling. Default is 3000.
        lower_threshold_mask (int, optional): Threshold for mask extraction. Default is 170.
        low_threshold (int, optional): Lower threshold for edge detection. Default is 0.
        high_threshold (int, optional): High threshold for edge detection. Default is 5.
        sampling_fraction (float, optional): Fraction of edge points to sample per slice.
        k_neighbors (int, optional): Number of neighbors for sampling.

    Returns:
        denoised_3d (np.ndarray): Denoised and windowed 3D image (uint8).
        mask_3d (np.ndarray): Corresponding 3D mask.
        sampled_point_cloud_3d (np.ndarray): (N, 5) array: x, y, z, grad, label.
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

    masked_slices = []
    sampled_point_cloud_3d = []

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
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        two_largest = sorted_contours[:2]
        selected_contours = []
        if two_largest:
            area1 = cv2.contourArea(two_largest[0])
            if area1 > 11000:
                selected_contours = [two_largest[0]]
            else:
                if len(two_largest) > 1:
                    selected_contours = two_largest
                else:
                    selected_contours = [two_largest[0]]
        logger.info(f"Detected {len(selected_contours)} contours in slice {slice_index+1}")
        shape = mask_uint8.shape[:2]
        filled_mask = Denoise.contour_to_filled_mask(selected_contours, shape)
        mask_uint8 = (filled_mask > 0).astype(np.uint8)
        mask_cv = mask_uint8 * 255
        slice_2d_uint8 = slice_2d.astype(np.uint8)
        masked_img = cv2.bitwise_and(slice_2d_uint8, slice_2d_uint8, mask=mask_cv)
        logger.info(f"Applied mask to denoised image for slice {slice_index+1}")
        edges_2d = Denoise.prewitt_2d(masked_img)
        edge_mask = edges_2d > high_threshold
        edge_points = np.argwhere(edge_mask)  # Each row: [x, y]

        # Build point cloud for this slice: (x, y, z, grad, label)
        slice_point_cloud = []
        for x, y in edge_points:
            grad = edges_2d[x, y]
            label = mask_2d[x, y]
            slice_point_cloud.append((x, y, slice_index, grad, label))

        num_points = len(slice_point_cloud)
        logger.info(f"Number of edge points in slice {slice_index+1}: {num_points}")

        # Perform slice-wise sampling
        if num_points > 0:
            points_xyz = np.array([p[:3] for p in slice_point_cloud], dtype=np.float32)
            grads = np.array([p[3] for p in slice_point_cloud], dtype=np.float32)
            labels = np.array([p[4] for p in slice_point_cloud], dtype=np.float32)
            pc = BasicPointCloud(points=points_xyz)
            sampled_pc, selected_indices = Denoise.curvature_preserving_sampling(
                pc, k_neighbors=k_neighbors, fraction=sampling_fraction
            )
            sampled_points = sampled_pc.points
            sampled_grads = grads[selected_indices]
            sampled_labels = labels[selected_indices]
            sampled_points_slice = np.column_stack((sampled_points, sampled_grads, sampled_labels))
            sampled_point_cloud_3d.append(sampled_points_slice)

        masked_slices.append(masked_img)

    denoised_3d = np.stack(masked_slices, axis=1)
    mask_3d = mask_data
    logger.info("Stacked all denoised slices into a 3D volume.")

    # Concatenate all sampled points from all slices
    if sampled_point_cloud_3d:
        sampled_point_cloud_3d = np.concatenate(sampled_point_cloud_3d, axis=0)
    else:
        sampled_point_cloud_3d = np.empty((0, 5), dtype=np.float32)

    # Save sampled point cloud
    if pointcloud_save_path is not None:
        np.save(pointcloud_save_path, sampled_point_cloud_3d)
        logger.info(f"Saved sampled 3D point cloud to {pointcloud_save_path}")

    return denoised_3d, mask_3d, sampled_point_cloud_3d

if __name__ == "__main__":
    denoised_3d, mask_3d, point_cloud_3d = three_d_denoising(
        cbct_path=r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\data.npy",
        mask_path=r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\gt_alpha.npy",
        pointcloud_save_path=r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled_6.npy", lower_threshold_mask= 150
    )
   
    # # Save the selected slices (or the entire denoised_img if that's what you want)
    # np.save(r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy", denoised_img)

    # denoised_img = np.load(r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy")
    # mask_3d = np.load(r"D:\Mphil\Programming\2DtoPointcloud_v2\dataset\P1-20250324T100522Z-001\P1\gt_alpha.npy")

    # pointcloud_formulation_3d(denoised_img, mask_3d, r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_sampled.npy", r"D:\Mphil\Programming\EfficientPointMed\output\3d\3d\pointcloud_npy.npy")
        