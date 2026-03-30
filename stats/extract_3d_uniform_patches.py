import numpy as np
import cv2

img = np.load(r"D:\Mphil\Programming\EfficientPointMed\dataset\P20\data.npy")
slice_no = 255
num_frames = 5  # successive frames

img_display = cv2.normalize(
    img[:, slice_no, :], None,
    alpha=0, beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)

# User selects ROI roughly (used just to get a point)
r = cv2.selectROI("Select ROI (10x10 will be enforced)", img_display, False)
print(f"Raw ROI from OpenCV: {r}")  # (x, y, w, h)

# Use the center of the user-selected ROI as anchor
cx = int(r[0] + r[2] / 2)
cy = int(r[1] + r[3] / 2)

# Enforce 10x10 window around the center
half = 5
x1, x2 = cx - half, cx + half
y1, y2 = cy - half, cy + half

# Clamp to image bounds
height, width = img.shape[0], img.shape[2]
x1 = max(0, min(x1, width - 10))
x2 = x1 + 10
y1 = max(0, min(y1, height - 10))
y2 = y1 + 10

print(f"Final fixed ROI (10x10): x({x1}:{x2}), y({y1}:{y2})")

# Create 3D block: (num_frames, 10, 10)
roi_block = np.zeros((num_frames, 10, 10), dtype=img.dtype)

for i in range(num_frames):
    slice_idx = slice_no + i
    if slice_idx < img.shape[1]:
        slice_img = img[:, slice_idx, :]
        roi_block[i] = slice_img[y1:y2, x1:x2]
    else:
        print(f"Warning: slice {slice_idx} out of bounds, leaving zeros")

out_path = rf"D:\Mphil\Dataset_overview\P20_stats\air_{slice_no}_3d.npy"
np.save(out_path, roi_block)
print(f"Saved 3D block: {roi_block.shape} -> {out_path}")

# Quick visual check
crop_display = cv2.normalize(roi_block[0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow("First 10x10 Cropped Slice", crop_display)
cv2.waitKey(0)
cv2.destroyAllWindows()


