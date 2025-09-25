# analyze_all_slices.py
# Auto-discover and analyze ALL slice .npy files under a root folder.
# For each group (parent folder) ➜ per-slice histogram PNGs, thumbnails (optional), Excel stats.
# Also emits a MASTER Excel concatenating stats across all groups.

from __future__ import annotations
import argparse
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import skew, kurtosis
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- core utils (from earlier script, trimmed to essentials) ----------

def robust_array(img: np.ndarray) -> np.ndarray:
    arr = np.array(img, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def compute_basic_stats(img: np.ndarray) -> dict:
    arr = robust_array(img).reshape(-1)
    finite = np.isfinite(arr)
    vals = arr[finite]
    n_total = arr.size
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(arr).sum())
    n_posinf = int(np.isposinf(arr).sum()) if hasattr(np, "isposinf") else 0
    n_neginf = int(np.isneginf(arr).sum()) if hasattr(np, "isneginf") else 0
    n_zeros = int((arr == 0).sum())

    if n_finite == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan, median=np.nan,
                    p01=np.nan, p05=np.nan, p25=np.nan, p50=np.nan, p75=np.nan, p95=np.nan, p99=np.nan,
                    dynamic_range=np.nan, skew=np.nan, kurtosis=np.nan,
                    n_total=n_total, n_finite=n_finite, n_nan=n_nan, n_posinf=n_posinf, n_neginf=n_neginf,
                    n_zeros=n_zeros, nonzero=n_total - n_zeros)

    s_min, s_max = float(vals.min()), float(vals.max())
    s_mean, s_std = float(vals.mean()), float(vals.std())
    s_med = float(np.median(vals))
    p01, p05, p25, p50, p75, p95, p99 = np.percentile(vals, [1,5,25,50,75,95,99]).astype(float)
    dyn = float(s_max - s_min)

    if _HAS_SCIPY:
        s_skew = float(skew(vals, bias=False))
        s_kurt = float(kurtosis(vals, fisher=True, bias=False))
    else:
        s = pd.Series(vals)
        s_skew = float(s.skew()) if not math.isnan(s.std()) else np.nan
        s_kurt = float(s.kurt()) if not math.isnan(s.std()) else np.nan

    return dict(min=s_min, max=s_max, mean=s_mean, std=s_std, median=s_med,
                p01=float(p01), p05=float(p05), p25=float(p25), p50=float(p50), p75=float(p75),
                p95=float(p95), p99=float(p99), dynamic_range=dyn,
                skew=s_skew, kurtosis=s_kurt,
                n_total=n_total, n_finite=n_finite, n_nan=n_nan, n_posinf=n_posinf, n_neginf=n_neginf,
                n_zeros=n_zeros, nonzero=n_total - n_zeros)

# def save_histogram_png(img: np.ndarray, out_path: Path, bins: int, clip_percent: float):
#     arr = robust_array(img).reshape(-1)
#     finite = np.isfinite(arr)
#     vals = arr[finite]
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     plt.figure(figsize=(6,4), dpi=150)
#     if vals.size == 0:
#         plt.title("No finite data")
#     else:
#         lo = np.percentile(vals, clip_percent)
#         hi = np.percentile(vals, 100 - clip_percent)
#         if hi <= lo:
#             lo, hi = float(vals.min()), float(vals.max())
#         data = vals[(vals >= lo) & (vals <= hi)]
#         plt.hist(data, bins=bins)
#         mu, med = float(vals.mean()), float(np.median(vals))
#         plt.axvline(mu)
#         plt.axvline(med)
#         plt.title(f"Histogram (clipped {clip_percent:.3g}% tails)")
#     plt.xlabel("Intensity"); plt.ylabel("Frequency"); plt.tight_layout()
#     plt.savefig(out_path); plt.close()

def save_histogram_png(img: np.ndarray, out_path: Path, bins: int, clip_percent: float,
                       draw_ref: bool = True,
                       mean_color: str = "tab:orange",
                       median_color: str = "tab:green"):
    """Save a histogram PNG, optionally with colored mean/median lines + legend."""
    arr = robust_array(img).reshape(-1)
    finite = np.isfinite(arr)
    vals = arr[finite]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4), dpi=150)

    if vals.size == 0:
        plt.title("No finite data")
    else:
        lo = np.percentile(vals, clip_percent)
        hi = np.percentile(vals, 100 - clip_percent)
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())

        data = vals[(vals >= lo) & (vals <= hi)]
        plt.hist(data, bins=bins)

        if draw_ref:
            mu, med = float(vals.mean()), float(np.median(vals))
            # colored, thin, and above bars
            plt.axvline(mu, linestyle="--", linewidth=1.2, color=mean_color, zorder=5, label=f"Mean = {mu:.2f}")
            # if mean==median, show just one line/label; else draw both
            if not np.isclose(mu, med):
                plt.axvline(med, linestyle="-.", linewidth=1.2, color=median_color, zorder=5, label=f"Median = {med:.2f}")
            plt.legend(loc="best")

        plt.title(f"Histogram (clipped {clip_percent:.3g}% tails)")

    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_png_thumbnail(img: np.ndarray, out_path: Path):
    from imageio import imwrite
    arr = robust_array(img)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        norm = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(arr, dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_path.as_posix(), norm)

def try_extract_index(path: Path) -> int | None:
    m = re.search(r"(\d+)(?=\D*$)", path.stem)
    return int(m.group(1)) if m else None

def plot_series_png(x, y, title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(7,4), dpi=150)
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title); plt.xlabel("Slice index"); plt.ylabel(ylabel); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path); plt.close()

# ---------- automation logic ----------

def parse_args():
    p = argparse.ArgumentParser(description="Auto analyze ALL 2D slice .npy files under a root.")
    p.add_argument("--root_in", required=True, help="Root folder to scan recursively for .npy slices.")
    p.add_argument("--out_root", required=True, help="Where to write analyses (per-group folders + master Excel).")
    p.add_argument("--pattern", default="*.npy", help="Glob pattern for files (default: *.npy).")
    p.add_argument("--group_depth", type=int, default=1,
                   help="Group by N-th parent of each file (1=immediate parent folder).")
    p.add_argument("--bins", type=int, default=256, help="Histogram bins.")
    p.add_argument("--clip_percent", type=float, default=0.5,
                   help="Percentile clipping on each tail for histograms.")
    p.add_argument("--save_png_thumb", action="store_true",
                   help="Also save normalized PNG thumbnails for each slice.")
    p.add_argument("--excel_name", default="slice_statistics.xlsx",
                   help="Per-group Excel filename.")
    p.add_argument("--master_excel", default="MASTER_slice_statistics.xlsx",
                   help="Name of the master Excel at out_root.")
    p.add_argument("--sort_by_index", action="store_true",
                   help="Sort per-group by numeric index in filename when available.")
    return p.parse_args()

def file_group_key(path: Path, depth: int) -> tuple[str, Path]:
    # depth=1 => parent.name ; depth=2 => parent.parent.name, etc.
    grp_parent = path.parents[depth-1] if depth >= 1 and len(path.parents) >= depth else path.parent
    return (grp_parent.name, grp_parent)

def ensure_2d_slices(npy: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """
    Accept either a 2D image or a 3D volume.
    - If 2D: return [(0, image)]
    - If 3D: try to guess slice axis; return list of (slice_idx, 2D slice)
    """
    if np.ndim(npy) == 2:
        return [(0, npy)]
    if np.ndim(npy) == 3:
        z, y, x = npy.shape
        # assume the smallest dimension is slices? Often shape (D,H,W).
        # Prefer a dimension that yields 2D slices with typical aspect (H,W).
        candidates = [(0, (z, y, x)), (1, (y, z, x)), (2, (x, z, y))]
        # default: if first axis produces 2D (y,x), use axis=0
        axis = 0
        # very simple heuristic: prefer axis where the remaining two dims are closer (not skinny).
        best = None; best_diff = None
        for ax in [0,1,2]:
            dims = [d for i,d in enumerate(npy.shape) if i != ax]
            diff = abs(dims[0] - dims[1])
            if (best is None) or (diff < best_diff):
                best, best_diff, axis = dims, diff, ax
        slices = []
        for i in range(npy.shape[axis]):
            if axis == 0:
                sl = npy[i, :, :]
            elif axis == 1:
                sl = npy[:, i, :]
            else:
                sl = npy[:, :, i]
            slices.append((i, sl))
        return slices
    # Other shapes unsupported
    raise ValueError(f"Expected 2D or 3D array, got shape {npy.shape}")

def analyze_group(files: list[Path], group_name: str, group_folder: Path,
                  out_root: Path, bins: int, clip_percent: float,
                  save_png_thumb: bool, excel_name: str, sort_by_index: bool) -> pd.DataFrame:
    print(f"\n[GROUP] {group_name}  ({len(files)} files)")
    out_dir = out_root / group_name
    plots_dir = out_dir / "per_slice_plots"
    thumbs_dir = out_dir / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if save_png_thumb:
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    series_x, series_mean, series_std, series_p25, series_p50, series_p75 = [], [], [], [], [], []

    # Optional sort by numeric index in filename
    if sort_by_index:
        files = sorted(files, key=lambda p: (try_extract_index(p) is None, try_extract_index(p), p.name))
    else:
        files = sorted(files)

    for f in files:
        try:
            arr = np.load(f)
        except Exception as e:
            print(f"[WARN] load failed: {f.name}: {e}")
            continue

        # Turn 3D into list of 2D slices if needed
        twods = ensure_2d_slices(arr)

        for loc_idx, img2d in twods:
            # choose slice index: numeric from filename, else loc_idx within volume
            idx_from_name = try_extract_index(f)
            slice_idx = idx_from_name if idx_from_name is not None else loc_idx

            stats = compute_basic_stats(img2d)
            rows.append({"group": group_name, "group_path": str(group_folder), "file": f.name,
                         "slice_local_index": loc_idx, "slice_index": slice_idx, **stats})

            # histogram + optional thumbnail
            stem = f"{Path(f.stem).name}_s{slice_idx}"
            save_histogram_png(img2d, plots_dir / f"{stem}_hist.png", bins, clip_percent)
            if save_png_thumb:
                save_png_thumbnail(img2d, thumbs_dir / f"{stem}.png")

            # aggregates
            series_x.append(slice_idx)
            series_mean.append(stats["mean"]); series_std.append(stats["std"])
            series_p25.append(stats["p25"]); series_p50.append(stats["p50"]); series_p75.append(stats["p75"])

    if not rows:
        print(f"[WARN] No analyzable slices for group {group_name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(by=["slice_index","file","slice_local_index"]).reset_index(drop=True)

    # write per-group Excel (slice_stats + summary)
    excel_path = out_dir / excel_name
    try:
        writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    except Exception:
        writer = pd.ExcelWriter(excel_path)

    df.to_excel(writer, sheet_name="slice_stats", index=False)

    # summary across this group
    summary = {}
    for col in ["mean","std","min","max","median","p01","p05","p25","p50","p75","p95","p99",
                "dynamic_range","skew","kurtosis","n_total","n_finite","n_nan","n_posinf","n_neginf","n_zeros","nonzero"]:
        if col in df.columns and df[col].notna().any():
            summary[f"{col}_mean_over_slices"] = df[col].mean()
            summary[f"{col}_median_over_slices"] = df[col].median()
            summary[f"{col}_min_over_slices"] = df[col].min()
            summary[f"{col}_max_over_slices"] = df[col].max()
    pd.DataFrame([summary]).to_excel(writer, sheet_name="summary_overall", index=False)
    writer.close()

    # aggregate PNG plots for the group
    order = np.argsort(np.array(series_x))
    xs = np.array(series_x)[order]
    plot_series_png(xs, np.array(series_mean)[order], "Mean by slice", "Mean", out_dir / "mean_by_slice.png")
    plot_series_png(xs, np.array(series_std)[order], "Std by slice", "Std", out_dir / "std_by_slice.png")
    plot_series_png(xs, np.array(series_p25)[order], "P25 by slice", "P25", out_dir / "p25_by_slice.png")
    plot_series_png(xs, np.array(series_p50)[order], "Median (P50) by slice", "P50", out_dir / "p50_by_slice.png")
    plot_series_png(xs, np.array(series_p75)[order], "P75 by slice", "P75", out_dir / "p75_by_slice.png")

    print(f"[OK] Group '{group_name}' ➜ {excel_path}")
    return df

def main():
    args = parse_args()
    root_in = Path(args.root_in).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # find all files matching pattern, recursively
    all_files = [p for p in root_in.rglob(args.pattern) if p.is_file()]
    if not all_files:
        raise SystemExit(f"No files found under {root_in} with pattern '{args.pattern}'")

    # group by parent depth
    groups = {}
    for f in all_files:
        gname, gpath = file_group_key(f, args.group_depth)
        groups.setdefault((gname, gpath), []).append(f)

    print(f"Discovered {len(all_files)} files across {len(groups)} group(s).")

    # analyze each group, accumulate master DF
    master_rows = []
    for (gname, gpath), files in sorted(groups.items(), key=lambda x: x[0][0]):
        df = analyze_group(files, gname, gpath, out_root,
                           args.bins, args.clip_percent,
                           args.save_png_thumb, args.excel_name, args.sort_by_index)
        if not df.empty:
            master_rows.append(df)

    if master_rows:
        master_df = pd.concat(master_rows, ignore_index=True)
        master_xlsx = out_root / args.master_excel
        master_df.to_excel(master_xlsx, index=False)
        print(f"\n[MASTER] Wrote consolidated stats: {master_xlsx}")
    else:
        print("\n[MASTER] No data written (no valid groups).")

if __name__ == "__main__":
    main()
