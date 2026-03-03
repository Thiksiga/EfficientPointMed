# hist_npy.py
# Usage:
#   python hist_npy.py --npy slice.npy --bins 256 --out hist.png
# Optional robust view (ignore extreme outliers in the P1..P99 range for the plot only):
#   python hist_npy.py --npy slice.npy --bins 512 --out hist.png --robust

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    #vmax = 6000

    print(f"Shape: {arr.shape}, dtype={arr.dtype}")
    print(f"Min={vmin:.6g}, Max={vmax:.6g}, Mean={arr.mean():.6g}, Std={arr.std():.6g}")
    print(f"P1={p1:.6g}, P99={p99:.6g}")

    # Choose range
    xr = (p1, p99) if args.robust else (vmin+1, vmax)


    #
    # plt.figure()
    # plt.hist(arr.ravel(), bins=args.bins, range=xr)
    # plt.title("Value Histogram")
    # plt.xlabel("Value")
    # plt.ylabel("Count")
    # plt.tight_layout()
    # plt.savefig(args.out, dpi=150)
    # plt.close()
    # print(f"Saved: {args.out}")

    plt.figure()
    # plt.hist(arr.ravel(), bins=args.bins, range=xr)
    sns.histplot(
        x=arr.ravel(),
        bins=args.bins,
        binrange=xr,  # xr should be a (min, max) tuple like 'range' in plt.hist
        kde=True,  # just histogram, no KDE
        stat="count"  # y-axis shows counts, like plt.hist
    )

    # Axis labels (Times New Roman, size 16)
    plt.xlabel("Value", fontname="Times New Roman", fontsize=16)
    plt.ylabel("Count", fontname="Times New Roman", fontsize=16)

    # Title (optional: also Times New Roman)
    plt.title("Value Histogram", fontname="Times New Roman", fontsize=16)

    # Tick labels (axis values): Times New Roman, size 14
    ax = plt.gca()

    # Mean and Std annotation (top-right corner)
    mean_val = arr.mean()
    std_val = arr.std()

    textstr = (
                  r"$\mu = %.3f$" "\n"
                  r"$\sigma = %.3f$"
              ) % (mean_val, std_val)

    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=14,
        fontname="Times New Roman",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(14)

    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(14)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
