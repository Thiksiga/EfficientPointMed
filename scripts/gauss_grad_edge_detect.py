# gauss_gradient_from_npy.py
import argparse
import numpy as np
import cv2
from math import ceil


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if radius is None:
        radius = int(ceil(3*sigma))
    x = np.arange(-radius, radius+1, dtype=np.float32)
    g = np.exp(-(x**2) / (2*sigma**2))
    g /= g.sum()
    return g.astype(np.float32)

def gaussian_derivative_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if radius is None:
        radius = int(ceil(3*sigma))
    x = np.arange(-radius, radius+1, dtype=np.float32)
    gd = -x * np.exp(-(x**2) / (2*sigma**2)) / (sigma**2)
    s = np.sum(np.abs(gd))
    if s > 0:
        gd /= s
    return gd.astype(np.float32)

def gauss_gradient_from_array(img2d: np.ndarray, sigma: float):
    if img2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img2d.shape}")
    gray = np.nan_to_num(img2d.astype(np.float32), copy=False)
    G  = gaussian_kernel1d(sigma)
    Gd = gaussian_derivative_kernel1d(sigma)
    Gx = cv2.sepFilter2D(gray, cv2.CV_32F, Gd, G)  # d/dx
    Gy = cv2.sepFilter2D(gray, cv2.CV_32F, G,  Gd) # d/dy
    mag = np.sqrt(Gx*Gx + Gy*Gy)                   # native units (e.g., HU/pixel)
    return Gx, Gy, mag

def non_max_suppression(Gx: np.ndarray, Gy: np.ndarray, mag: np.ndarray) -> np.ndarray:
    ang = (np.degrees(np.arctan2(Gy, Gx)) + 180.0) % 180.0
    h, w = mag.shape
    Z = np.zeros_like(mag, dtype=np.float32)
    for y in range(1, h-1):
        a_row = ang[y]; m_row = mag[y]
        for x in range(1, w-1):
            a = a_row[x]; m = m_row[x]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                q, r = mag[y, x+1], mag[y, x-1]
            elif 22.5 <= a < 67.5:
                q, r = mag[y+1, x-1], mag[y-1, x+1]
            elif 67.5 <= a < 112.5:
                q, r = mag[y+1, x], mag[y-1, x]
            else:
                q, r = mag[y-1, x-1], mag[y+1, x+1]
            Z[y, x] = m if (m >= q and m >= r) else 0.0
    return Z

def threshold_edges(mag: np.ndarray, abs_threshold: float | None, percentile: float | None):
    if abs_threshold is None and percentile is None:
        return None
    t = float(abs_threshold) if abs_threshold is not None else np.percentile(mag, percentile)
    return (mag >= t).astype(np.uint8) * 255

def main():
    ap = argparse.ArgumentParser(description="Gauss-gradient edge detection on a 2D .npy slice (native intensity).")
    ap.add_argument("--npy", required=True, help="Path to 2D NumPy array (.npy), e.g., HU in [-1000, 3000].")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian sigma in pixels.")
    ap.add_argument("--nms", action="store_true", help="Apply non-maximum suppression (thinning).")
    ap.add_argument("--abs_thresh", type=float, default=None, help="Absolute threshold on |∇I| in native units.")
    ap.add_argument("--pct", type=float, default=None, help="Percentile threshold on |∇I| (e.g., 95 keeps top 5%).")
    ap.add_argument("--save_mag_npy", default=None, help="Path to save gradient magnitude as .npy (native units).")
    ap.add_argument("--save_edges_png", default=None, help="Path to save binary edges PNG (0/255).")
    args = ap.parse_args()

    arr = np.load(args.npy)  # <<— THIS replaces cv2.imread
    Gx, Gy, mag = gauss_gradient_from_array(arr, sigma=args.sigma)
    if args.nms:
        mag = non_max_suppression(Gx, Gy, mag)

    if args.save_mag_npy:
        np.save(args.save_mag_npy, mag)

    if args.abs_thresh is not None or args.pct is not None:
        edges = threshold_edges(mag, args.abs_thresh, args.pct)
        if args.save_edges_png:
            cv2.imwrite(args.save_edges_png, edges)
    else:
        edges = None

    # Simple printouts for quick checks
    print(f"Loaded {arr.shape}, dtype={arr.dtype}, min={arr.min():.2f}, max={arr.max():.2f}")
    print(f"Gradient magnitude: min={mag.min():.4f}, max={mag.max():.4f}")
    if edges is not None:
        print(f"Edges nonzero: {int((edges>0).sum())}")

if __name__ == "__main__":
    main()
