# visualize_npy_matplotlib.py
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Image (H, W)
# ---------------------------
def visualize_image(arr2d: np.ndarray, title: str = "Image"):
    if arr2d.ndim != 2:
        raise ValueError(f"visualize_image expects a 2D array, got shape {arr2d.shape}")
    plt.figure(figsize=(7, 6))
    plt.imshow(arr2d, cmap="gray")
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ---------------------------
# Point cloud (N,5): x,y,z,grad,label  -> use x,y only
# ---------------------------
def visualize_xy_from_n5(arr_n5: np.ndarray,
                         color_by: str = "label",    # "label" or "grad"
                         point_size: float = 2.0,
                         alpha: float = 0.9,
                         cmap: str = "viridis",
                         flip_y: bool = False,
                         title: str = "2D slice (x,y)"):
    if arr_n5.ndim != 2 or arr_n5.shape[1] != 5:
        raise ValueError(f"Expected (N,5) [x,y,z,grad,label], got {arr_n5.shape}")

    x = arr_n5[:, 0].astype(float)
    y = arr_n5[:, 1].astype(float)
    grad = arr_n5[:, 3].astype(float)
    label = arr_n5[:, 4]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if color_by.lower() == "label":
        mask1 = label >= 0.5
        # label==0
        ax.scatter(x[~mask1], y[~mask1], s=point_size, c="#4CD3F5",
                   alpha=alpha, edgecolors="none", marker=".", label="label=0", rasterized=True)
        # label==1
        ax.scatter(x[mask1], y[mask1], s=point_size, c="#E74C3C",
                   alpha=alpha, edgecolors="none", marker=".", label="label=1", rasterized=True)
        ax.legend(loc="best", frameon=False)
    else:  # grad
        g = grad.copy()
        gmin, gmax = np.nanmin(g), np.nanmax(g)
        denom = gmax - gmin if gmax > gmin else 1.0
        gnorm = (g - gmin) / denom
        sc = ax.scatter(x, y, s=point_size, c=gnorm, cmap=cmap,
                        alpha=alpha, edgecolors="none", marker=".", rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("grad (normalized)")

    if flip_y:
        ax.invert_yaxis()

    # Tight view
    pad = 0.02
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)

    plt.tight_layout()
    plt.show()

# ---------------------------
# Auto-dispatch
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Visualize a .npy file as image (H,W) or as 2D XY scatter from (N,5)=(x,y,z,grad,label)."
    )
    ap.add_argument("--npy", required=True, help="Path to .npy file")
    ap.add_argument("--force", choices=["auto", "image", "pc"], default="auto",
                    help="Force interpretation mode (default: auto)")
    ap.add_argument("--color_by", choices=["label", "grad"], default="label",
                    help="For point clouds: color by label or grad")
    ap.add_argument("--point_size", type=float, default=2.0, help="Matplotlib marker size")
    ap.add_argument("--alpha", type=float, default=0.9, help="Point alpha (0..1)")
    ap.add_argument("--cmap", default="viridis", help="Colormap when color_by=grad")
    ap.add_argument("--flip_y", action="store_true", help="Invert Y axis (image-like coordinates)")
    ap.add_argument("--title", default=None, help="Plot title override")
    ap.add_argument("--save", default=None, help="Optional path to save the figure (e.g., .png)")
    args = ap.parse_args()

    arr = np.load(args.npy, allow_pickle=False)
    print("Loaded:", args.npy, "| shape:", arr.shape, "| dtype:", arr.dtype)

    mode = args.force
    if mode == "auto":
        if arr.ndim == 2 and arr.shape[1] == 5:
            mode = "pc"
        elif arr.ndim == 2:
            mode = "image"
        else:
            raise ValueError(f"Unsupported shape {arr.shape}. Expected (H,W) or (N,5).")

    if mode == "image":
        visualize_image(arr, title=args.title or f"Image: {args.npy}")
    else:
        # For point cloud: (N,5)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"Forced 'pc' but got shape {arr.shape}; need (N,5).")
        visualize_xy_from_n5(
            arr,
            color_by=args.color_by,
            point_size=args.point_size,
            alpha=args.alpha,
            cmap=args.cmap,
            flip_y=args.flip_y,
            title=args.title or f"XY from (N,5): {args.npy}",
        )
        if args.save:
            plt.savefig(args.save, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()

###---CMD----###
# --python 2d_npy_pc.py --npy "D:\Mphil\Programming\EfficientPointMed\output\3d\pc_test_slice\slices\sampled_points_slice111.npy" --color_by label --point_size 2