import numpy as np
import matplotlib.pyplot as plt
import re

file_path = rf"C:\Users\thiks\Downloads\reconstructed_volume.npy"
    # Load the array from the .npy file
arr = np.load(file_path)

    # Print the array
print(arr.shape)

d = arr.shape[1]

for i in range(0,d):
    print(i)
    # Replace this with your actual file path
    # file_path = rf"D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\sampled_points_slice\sampled_points_slice_{i}.npy"

    # match = re.search(r"_(\d+)\.npy$", file_path)
    # if match:
    #     idx = int(match.group(1))
    #     print("Extracted idx:", idx)
    # else:
    #     print("No index found in path.")

    # Save the 2D array as an image
    # plt.imsave(rf'D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\img\slice_{idx}.png', arr, cmap='gray')
    plt.imsave(rf'C:\Users\thiks\Downloads\reconstructed_p10_MSEnoiseLoss_TV_noiseJS_50_0.05rem_60_coronal_halfPrecision\reconstructed_p10_32_{i}.png', arr[:, i,:], cmap='gray')

# num_slices = arr.shape[1]
# slice_indices = [0, num_slices // 2, num_slices - 1]
# selected_slices = arr[:, slice_indices, :]s

# for i, idx in enumerate(slice_indices):
#     plt.imsave(rf'D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\img\slice_{idx}.png', arr[:,idx,:], cmap='gray')
#     print(f"Saved slice_{idx}.png")