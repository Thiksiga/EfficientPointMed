import matplotlib.pyplot as plt
import numpy as np

def visualize_image(arr2d: np.ndarray, title: str = "Image"):
    if arr2d.ndim != 2:
        raise ValueError(f"visualize_image expects a 2D array, got shape {arr2d.shape}")
    plt.figure(figsize=(7, 6))
    plt.imshow(arr2d, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img = np.load(r"D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\slices\denoised_masked195.npy")

    visualize_image(img, title="Image")