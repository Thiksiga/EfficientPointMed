import numpy as np
import matplotlib.pyplot as plt

data_original = np.load("../dataset/P1-20250324T100522Z-001/P1/data.npy")
data_denoised = np.load("../output/denoise_dev/inv_log_177.npy")

print("Shape of the original numpy array: ", data_original[:,176,:].shape)
print("Shape of the denoised numpy array: ", data_denoised.shape)

# print("original image unique values: ", np.unique(data_original[:,176,:]))
# print("denoised image unique values: ", np.unique(data_denoised))

data_original_flatten = data_original.flatten()
data_denoised_flatten = data_denoised.flatten()

fig, ax = plt.subplots(figsize=(6, 4))
ax.violinplot([data_original_flatten, data_denoised_flatten],
              showmeans=False,
              showmedians=True,
              showextrema=True,
              quantiles=[[0.25, 0.75], [0.25, 0.75]])
ax.set_xticks([1, 2], ("original", "denoised"))
ax.set_ylabel("Intensity")
ax.set_title("Intensity distribution")
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


