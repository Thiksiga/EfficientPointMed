# # viz_o3d_min.py
# import numpy as np, open3d as o3d, sys
#
# # path = sys.argv[1] if len(sys.argv) > 1 else "points.npy"
# path = r"D:\Mphil\Programming\EfficientPointMed\output\denoise_dev\sampled_points_slice_177.npy"
# arr = np.load(path)
#
#
# if arr.ndim != 2 or arr.shape[1] < 4:
#     raise ValueError(f"Expected (N,5), got {arr.shape}")
#
# # Keep only finite rows
# m = np.isfinite(arr).all(axis=1)
# arr = arr[m]
# if arr.size == 0:
#     raise ValueError("No finite points in file after filtering.")
#
# xyz = arr[:, :3].astype(np.float32)
# grad = arr[:, 3].astype(np.float32)
# label = arr[:, 4]
#
# # ---- Print quick diagnostics
# mins = xyz.min(0); maxs = xyz.max(0); span = maxs - mins
# print(f"N={xyz.shape[0]}  min={mins}  max={maxs}  span={span}")
# print(f"Any NaNs? {np.isnan(xyz).any()}  Any Infs? {np.isinf(xyz).any()}")
#
# # ---- Normalize to unit cube centered at origin (for robust viewing)
# eps = 1e-12
# center = (mins + maxs) / 2.0
# scale = float(np.max(span)) + eps
# xyz_norm = (xyz - center) / scale
#
# # ---- Color by gradient (grayscale, robust percentiles)
# p2, p98 = np.percentile(grad, [2, 98])
# g = (grad - p2) / max(p98 - p2, 1e-6)
# g = np.clip(g, 0, 1).astype(np.float32)
# colors = np.stack([g, g, g], 1)
#
# # If all grayscale near 0 → switch to white so it’s visible on black bg
# if (colors.mean() < 0.05):
#     colors[:] = 1.0
#
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz_norm)
# pcd.colors = o3d.utility.Vector3dVector(colors)
#
# # Simple viewer that auto-fits
# o3d.visualization.draw_geometries(
#     [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
#     window_name=f"Open3D: {path}",
#     width=1280, height=800,
#     point_show_normal=False
# )


###-------------For 2d sliced view------------------------------###
#!/usr/bin/env python3
# viz_xy_matplotlib.py
# Usage:
#   python viz_xy_matplotlib.py --npy path/to/points.npy --color-by grad
#   python viz_xy_matplotlib.py --npy path/to/points.npy --color-by label --invert-y
#   python viz_xy_matplotlib.py --npy path/to/points.npy --s 3 --alpha 0.8 --save out.png

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="2D scatter of N×5 [x,y,z,grad,label] (z omitted).")
    p.add_argument("--npy", required=True, help="Path to .npy with shape (N,5).")
    p.add_argument("--color-by", choices=["grad", "label", "none"], default="grad",
                   help="Color points by gradient, label, or none.")
    p.add_argument("--s", type=float, default=2.0, help="Marker size.")
    p.add_argument("--alpha", type=float, default=0.9, help="Marker alpha.")
    p.add_argument("--invert-y", action="store_true", help="Invert Y axis (image row->Cartesian).")
    p.add_argument("--save", type=str, default=None, help="Optional path to save PNG.")
    p.add_argument("--title", type=str, default=None, help="Optional plot title.")
    return p.parse_args()

def robust_norm(v):
    v = np.asarray(v, dtype=np.float32)
    p2, p98 = np.percentile(v, [2, 98])
    if p98 > p2:
        v = (v - p2) / (p98 - p2)
    else:
        v = (v - v.min()) / max(v.ptp(), 1e-12)
    return np.clip(v, 0.0, 1.0)

def main():
    args = parse_args()
    arr = np.load(args.npy)

    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected shape (N,5), got {arr.shape}")

    # Keep only finite rows
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.size == 0:
        raise ValueError("No finite rows after filtering.")

    x, y = arr[:, 0].astype(np.float32), arr[:, 1].astype(np.float32)
    grad   = arr[:, 2].astype(np.float32)
    labels = arr[:, 3]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=110)

    if args.color_by == "grad":
        c = robust_norm(grad)
        sc = ax.scatter(x, y, c=c, s=args.s, alpha=args.alpha, cmap="viridis", rasterized=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Gradient (robust norm)")
        ax.set_title(args.title or "XY scatter colored by grad")
    elif args.color_by == "label":
        uniq = np.unique(labels.astype(int))
        # Use a categorical colormap
        cmap = plt.get_cmap("tab20", max(len(uniq), 1))
        for i, u in enumerate(uniq):
            sel = (labels.astype(int) == u)
            ax.scatter(x[sel], y[sel], s=args.s, alpha=args.alpha,
                       color=cmap(i), label=f"label={u}", rasterized=True)
        if len(uniq) <= 25:
            ax.legend(loc="best", frameon=True, fontsize=9)
        ax.set_title(args.title or "XY scatter colored by label")
    else:
        ax.scatter(x, y, s=args.s, alpha=args.alpha, color="black", rasterized=True)
        ax.set_title(args.title or "XY scatter")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    if args.invert_y:
        ax.invert_yaxis()

    # Tight limits with small padding
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    pad_x = max(1e-6, 0.02 * (xmax - xmin if xmax > xmin else 1.0))
    pad_y = max(1e-6, 0.02 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")
        print(f"Saved figure to: {args.save}")

    plt.show()

if __name__ == "__main__":
    main()
