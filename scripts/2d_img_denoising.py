import numpy as np
import matplotlib.pyplot as plt
import cv2
from scripts.denoise_utilitiess import Denoise, BasicPointCloud
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load your 3D CBCT and mask data
cbct_data = np.load("../dataset/P1-20250324T100522Z-001/P1/data.npy")    # shape: (z, y, x) or similar
mask_data = np.load("../dataset/P1-20250324T100522Z-001/P1/gt_alpha.npy") # shape: (z, y, x) or similar

# Select the 176th slice along the 2nd axis (y-axis)
slice_idx = 176
hu_min = -1000
hu_max = 4000
lower_threshold_mask = 1000
high_threshold = 5
k_neighbors = 5
sampling_fraction = 0.5

cbct_slice = cbct_data[:, slice_idx, :]        # shape: (z, x)
mask_slice = mask_data[:, slice_idx, :]        # shape: (z, x)

# Hounsfield to uint8
# converted = Denoise.hounsfield_to_255(cbct_data, hu_min, hu_max)

# # Normalize image for display
# cbct_norm = (cbct_slice - cbct_slice.min()) / (cbct_slice.max() - cbct_slice.min())

output = Denoise.adaptive_median_filter(cbct_slice, 7)
logger.info(f"Applied adaptive median filter to slice {slice_idx+1}")
np.save(f"../output/denoise_dev/median_{slice_idx+1}.npy", output)

log_transformed, original_min, eps = Denoise.log_transform_for_denoising(output)
logger.info(f"Applied log transform to slice {slice_idx+1}")
np.save(f"../output/denoise_dev/log_transformed_{slice_idx+1}.npy", log_transformed)

denoised_log = Denoise.denoise_log_transformed_image(log_transformed)
logger.info(f"Denoised log-transformed image for slice {slice_idx+1}")
np.save(f"../output/denoise_dev/denoised_log_{slice_idx+1}.npy", denoised_log)

inv_log = Denoise.inverse_log_transform(denoised_log, original_min, eps)
logger.info(f"Inverse log transform applied to slice {slice_idx+1}")
np.save(f"../output/denoise_dev/inv_log_{slice_idx+1}.npy", inv_log)

# denoised_img_uint8 = Denoise.apply_window(inv_log, window_center=40, window_width=400)
# logger.info(f"Applied windowing to slice {slice_idx+1}")
# np.save(f"../output/denoise_dev/denoised_img_{slice_idx+1}.npy", denoised_img_uint8)

_, mask = cv2.threshold(inv_log, lower_threshold_mask, 1, cv2.THRESH_BINARY)
print("mask: ",np.unique(mask))
logger.info(f"Applied thresholding to slice {slice_idx+1}")
np.save(f"../output/denoise_dev/masked_init_{slice_idx+1}.npy", mask)
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
logger.info(f"Detected {len(selected_contours)} contours in slice {slice_idx+1}")
# np.save(f"../output/denoise_dev/contours_{slice_idx+1}.npy", selected_contours)
shape = mask_uint8.shape[:2]
filled_mask = Denoise.contour_to_filled_mask(selected_contours, shape)
# mask_uint8 = (filled_mask > 0).astype(np.uint8)
# mask_cv = mask_uint8 * 255
print("filled+mask_unique: ",np.unique(filled_mask))
np.save(f"../output/denoise_dev/filled_mask_{slice_idx+1}.npy", filled_mask)

# slice_2d_uint8 = slice_2d.astype(np.uint8)
masked_img = cv2.bitwise_and(inv_log, inv_log, mask=filled_mask)
logger.info(f"Applied mask to denoised image for slice {slice_idx+1}")
np.save(f"../output/denoise_dev/masked_img_{slice_idx+1}.npy", masked_img)


#-----------------Avoid below code for now-------------
edges_2d = Denoise.prewitt_2d(masked_img)
print("edges_2d_unique:",np.unique(edges_2d))
np.save(f"../output/denoise_dev/edges_2d_{slice_idx+1}.npy", edges_2d)
edge_mask = edges_2d > high_threshold
np.save(f"../output/denoise_dev/edge_mask_{high_threshold}_{slice_idx+1}.npy", edge_mask)
edge_points = np.argwhere(edge_mask)  # Each row: [x, y]
print("edge_points: ",edge_points)

# Build point cloud for this slice: (x, y, z, grad, label)
slice_point_cloud = []
for x, y in edge_points:
    grad = edges_2d[x, y]
    label = mask_slice[x, y]
    slice_point_cloud.append((x, y, slice_idx, grad, label))
    # slice_point_cloud.append((x, y, grad, label))

num_points = len(slice_point_cloud)
logger.info(f"Number of edge points in slice {slice_idx+1}: {num_points}")
np.save(f"../output/denoise_dev/slice_point_cloud_{slice_idx+1}.npy", slice_point_cloud)

sampled_point_cloud_3d = []
masked_slices = []

# Perform slice-wise sampling
if num_points > 0:
    # points_xyz = np.array([p[:3] for p in slice_point_cloud], dtype=np.float32)
    # grads = np.array([p[3] for p in slice_point_cloud], dtype=np.float32)
    # labels = np.array([p[4] for p in slice_point_cloud], dtype=np.float32)
    points = np.array([(x, y, z) for (x, y, z, grad, label) in slice_point_cloud])
    labels = np.array([label for (x, y, z, grad, label) in slice_point_cloud])
    pc = BasicPointCloud(points=points)
    sampled_pc, selected_indices = Denoise.curvature_preserving_sampling(
        pc, k_neighbors=k_neighbors, fraction=sampling_fraction
    )
    sampled_points = sampled_pc.points
    sampled_labels = labels[selected_indices]
    sampled_points_slice = np.column_stack((sampled_points, sampled_labels))
    np.save(f"../output/denoise_dev/sampled_points_slice_{slice_idx + 1}.npy", sampled_points_slice)
    sampled_point_cloud_3d.append(sampled_points_slice)

# masked_slices.append(masked_img)
# np.save(f"../output/denoise_dev/masked_slices_1.npy", masked_slices)









