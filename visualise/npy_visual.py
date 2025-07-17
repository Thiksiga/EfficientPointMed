import numpy as np
import matplotlib.pyplot as plt

# Replace this with your actual file path
file_path = r"D:\Mphil\Programming\EfficientPointMed\output\selected_denoised_3d.npy"

# Load the array from the .npy file
arr = np.load(file_path)

# Print the array
print(arr.shape)

num_slices = arr.shape[1]
slice_indices = [0, num_slices // 2, num_slices - 1]
selected_slices = arr[:, slice_indices, :]

for i, idx in enumerate(slice_indices):
    plt.imsave(f'slice_{idx}.png', arr[:,idx,:], cmap='gray')
    print(f"Saved slice_{idx}.png")