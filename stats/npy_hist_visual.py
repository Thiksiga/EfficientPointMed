# hist_npy.py
# Usage:
#   python hist_npy.py --npy slice.npy --bins 256 --out hist.png
# Optional robust view (ignore extreme outliers in the P1..P99 range for the plot only):
#   python hist_npy.py --npy slice.npy --bins 512 --out hist.png --robust

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot histogram of values in a .npy array.")
    ap.add_argument("--npy", required=True, help="Path to .npy file (any shape).")
    ap.add_argument("--bins", type=int, default=256, help="Number of histogram bins.")
    ap.add_argument("--out",  default="hist.png", help="Output PNG path.")
    ap.add_argument("--robust", action="store_true",
                    help="If set, plot x-range from 1st to 99th percentile (data unchanged).")
    args = ap.parse_args()

    arr = np.load(args.npy)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, copy=False)

    vmin, vmax = float(arr.min()), float(arr.max())
    p1, p99 = np.percentile(arr, [1, 99])

    unq_val = np.unique(arr)

    print(unq_val.shape)

    vmax = 3000

    print(f"Shape: {arr.shape}, dtype={arr.dtype}")
    print(f"Min={vmin:.6g}, Max={vmax:.6g}, Mean={arr.mean():.6g}, Std={arr.std():.6g}")
    print(f"P1={p1:.6g}, P99={p99:.6g}")

    # Choose range
    xr = (p1, p99) if args.robust else (vmin+1, vmax)

    plt.figure()
    plt.hist(arr.ravel(), bins=args.bins, range=xr)
    plt.title("Value Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
