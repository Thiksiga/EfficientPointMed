import numpy as np
import matplotlib.pyplot as plt
import re

# Replace this with your actual file path
file_path = r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy"

# Load the array from the .npy file
arr = np.load(file_path)

# Print the array
print(arr.shape)

# match = re.search(r"_(\d+)\.npy$", file_path)
# if match:
#     idx = int(match.group(1))
#     print("Extracted idx:", idx)
# else:
#     print("No index found in path.")

# Save the 2D array as an image
# plt.imsave(rf'D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\img\slice_{idx}.png', arr, cmap='gray')
plt.imsave(rf'D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\img\edges_foreground.png', arr, cmap='gray')

# num_slices = arr.shape[1]
# slice_indices = [0, num_slices // 2, num_slices - 1]
# selected_slices = arr[:, slice_indices, :]

# for i, idx in enumerate(slice_indices):
#     plt.imsave(rf'D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\img\slice_{idx}.png', arr[:,idx,:], cmap='gray')
#     print(f"Saved slice_{idx}.png")