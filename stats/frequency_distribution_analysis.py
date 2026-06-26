import numpy as np
import matplotlib.pyplot as plt

# # load CBCT volume
# img = np.load(r"D:\Mphil\Dataset_overview\P50_stats\bony_200_3d.npy")
#
# # select a slice
# slice_img = img[img.shape[0]//2]
#
# # compute Fourier transform
# f = np.fft.fft2(slice_img)
#
# # shift zero frequency to center
# fshift = np.fft.fftshift(f)
#
# # magnitude spectrum
# magnitude = np.log(np.abs(fshift) + 1)
#
# plt.figure(figsize=(10,4))
#
# plt.subplot(1,2,1)
# plt.title("Original Slice")
# plt.imshow(slice_img, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1,2,2)
# plt.title("Frequency Spectrum")
# plt.imshow(magnitude, cmap='gray')
# plt.axis('off')
#
# plt.show()

###--------Radial power spectrum------------------
import numpy as np
import matplotlib.pyplot as plt

slice_img = img[img.shape[0]//2]

f = np.fft.fft2(slice_img)
fshift = np.fft.fftshift(f)

power = np.abs(fshift)**2

h, w = power.shape
center = (h//2, w//2)

y, x = np.indices((h, w))
r = np.sqrt((x-center[1])**2 + (y-center[0])**2)
r = r.astype(np.int32)

tbin = np.bincount(r.ravel(), power.ravel())
nr = np.bincount(r.ravel())

radial_profile = tbin / nr

plt.plot(radial_profile)
plt.title("Radial Frequency Distribution")
plt.xlabel("Frequency radius")
plt.ylabel("Power")
plt.show()

###--------Wavelet analysis----------------
"""
3D Medical Volume Wavelet Analysis Script
----------------------------------------

Performs multilevel 3D discrete wavelet decomposition on a grayscale medical volume,
then computes statistical summaries, subband energies, coefficient histograms,
and representative slice visualizations.

Supported inputs:
- .npy
- .nii / .nii.gz (requires nibabel)

Example usage:
--------------
python wavelet_3d_analysis.py \
    --input "D:/data/scan.npy" \
    --output_dir "D:/data/wavelet_results" \
    --wavelet db4 \
    --level 3 \
    --mode symmetric \
    --bins 100 \
    --save_coeffs

Optional:
---------
--clip_min -1000
--clip_max 5646
--use_abs_for_hist
"""

import os
import json
import math
import argparse
import warnings
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from scipy.stats import skew, kurtosis

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def robust_sigma_mad(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return mad / 0.6745 if mad != 0 else 0.0


def entropy_from_hist(values: np.ndarray, bins: int = 100) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan

    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    p = hist / np.sum(hist)
    return -np.sum(p * np.log2(p))


def subband_energy(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    return float(np.sum(arr ** 2))


def sparsity_ratio(arr: np.ndarray, threshold: float = 1e-6) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(np.abs(arr) < threshold))


def get_middle_slices(volume: np.ndarray) -> Dict[str, np.ndarray]:
    z, y, x = volume.shape
    return {
        "axial": volume[z // 2, :, :],
        "coronal": volume[:, y // 2, :],
        "sagittal": volume[:, :, x // 2]
    }


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    finite_mask = np.isfinite(img)
    if not np.any(finite_mask):
        return np.zeros_like(img, dtype=np.float32)

    vals = img[finite_mask]
    vmin = np.percentile(vals, 1)
    vmax = np.percentile(vals, 99)

    if vmax <= vmin:
        vmax = np.max(vals)
        vmin = np.min(vals)

    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)

    out = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return out


def plot_single_slice(
    img: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "gray"
) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_histogram(
    values: np.ndarray,
    bins: int,
    title: str,
    save_path: str
) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("Coefficient value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_energy_bar(df: pd.DataFrame, save_path: str) -> None:
    if df.empty:
        return

    labels = [f"L{lvl}_{band}" for lvl, band in zip(df["level"], df["subband"])]
    energies = df["energy"].values

    plt.figure(figsize=(max(10, len(labels) * 0.45), 5))
    plt.bar(labels, energies)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Energy")
    plt.title("Wavelet Subband Energy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Input loading
# -----------------------------
def load_volume(input_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext_lower = input_path.lower()

    metadata = {
        "input_path": input_path,
        "spacing": None,
        "affine": None,
        "source_type": None,
    }

    if ext_lower.endswith(".npy"):
        volume = np.load(input_path)
        metadata["source_type"] = "npy"

    elif ext_lower.endswith(".nii") or ext_lower.endswith(".nii.gz"):
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required to read NIfTI files. "
                "Install it with: pip install nibabel"
            )
        nii = nib.load(input_path)
        volume = nii.get_fdata()
        metadata["source_type"] = "nifti"
        metadata["spacing"] = tuple(float(x) for x in nii.header.get_zooms()[:3])
        metadata["affine"] = nii.affine.tolist()

    else:
        raise ValueError(
            "Unsupported input format. Please provide a .npy, .nii, or .nii.gz file."
        )

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, but got shape {volume.shape}")

    return volume.astype(np.float32), metadata


# -----------------------------
# Wavelet analysis
# -----------------------------
def analyze_subband(
    coeff_array: np.ndarray,
    level: int,
    subband_name: str,
    total_energy_ref: float,
    bins: int = 100,
    use_abs_for_hist: bool = False
) -> Dict[str, Any]:
    arr = coeff_array.astype(np.float32)
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        return {
            "level": level,
            "subband": subband_name,
            "shape": tuple(arr.shape),
            "num_coeffs": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "variance": np.nan,
            "median": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "l1_norm": np.nan,
            "l2_norm": np.nan,
            "energy": np.nan,
            "relative_energy": np.nan,
            "mad_sigma": np.nan,
            "entropy": np.nan,
            "sparsity_ratio_1e-6": np.nan,
        }

    energy = subband_energy(flat)
    relative_energy = energy / total_energy_ref if total_energy_ref > 0 else np.nan

    hist_values = np.abs(flat) if use_abs_for_hist else flat

    row = {
        "level": level,
        "subband": subband_name,
        "shape": tuple(arr.shape),
        "num_coeffs": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "variance": float(np.var(flat)),
        "median": float(np.median(flat)),
        "skewness": float(skew(flat, bias=False)) if flat.size > 2 else np.nan,
        "kurtosis": float(kurtosis(flat, fisher=True, bias=False)) if flat.size > 3 else np.nan,
        "l1_norm": float(np.sum(np.abs(flat))),
        "l2_norm": float(np.sqrt(np.sum(flat ** 2))),
        "energy": float(energy),
        "relative_energy": float(relative_energy),
        "mad_sigma": float(robust_sigma_mad(flat)),
        "entropy": float(entropy_from_hist(hist_values, bins=bins)),
        "sparsity_ratio_1e-6": float(sparsity_ratio(flat, threshold=1e-6)),
    }
    return row


def run_wavelet_analysis(
    volume: np.ndarray,
    wavelet_name: str,
    level: int,
    mode: str,
    bins: int,
    output_dir: str,
    save_coeffs: bool,
    use_abs_for_hist: bool
) -> pd.DataFrame:
    coeffs = pywt.wavedecn(volume, wavelet=wavelet_name, level=level, mode=mode)

    coeff_root = os.path.join(output_dir, "coefficients")
    hist_root = os.path.join(output_dir, "histograms")
    slice_root = os.path.join(output_dir, "slices")
    ensure_dir(coeff_root)
    ensure_dir(hist_root)
    ensure_dir(slice_root)

    total_energy_ref = float(np.sum(volume.astype(np.float64) ** 2))

    rows = []

    # coeffs[0] is approximation at final level
    approx = coeffs[0]
    approx_name = f"LLL_level_{level}"

    row = analyze_subband(
        coeff_array=approx,
        level=level,
        subband_name=approx_name,
        total_energy_ref=total_energy_ref,
        bins=bins,
        use_abs_for_hist=use_abs_for_hist
    )
    rows.append(row)

    if save_coeffs:
        np.save(os.path.join(coeff_root, f"{approx_name}.npy"), approx)

    plot_histogram(
        np.abs(approx).ravel() if use_abs_for_hist else approx.ravel(),
        bins=bins,
        title=f"Histogram: {approx_name}",
        save_path=os.path.join(hist_root, f"{approx_name}_hist.png")
    )

    for plane_name, sl in get_middle_slices(approx).items():
        plot_single_slice(
            normalize_for_display(sl),
            title=f"{approx_name} - {plane_name}",
            save_path=os.path.join(slice_root, f"{approx_name}_{plane_name}.png")
        )

    # coeffs[1:] are detail dicts from highest approximation level downward
    # PyWavelets order: coeffs[1] corresponds to level=final_level, then final_level-1, ...
    for i, detail_dict in enumerate(coeffs[1:]):
        current_level = level - i

        for band_name, band_arr in detail_dict.items():
            label = f"{band_name}_level_{current_level}"

            row = analyze_subband(
                coeff_array=band_arr,
                level=current_level,
                subband_name=band_name,
                total_energy_ref=total_energy_ref,
                bins=bins,
                use_abs_for_hist=use_abs_for_hist
            )
            rows.append(row)

            if save_coeffs:
                np.save(os.path.join(coeff_root, f"{label}.npy"), band_arr)

            plot_histogram(
                np.abs(band_arr).ravel() if use_abs_for_hist else band_arr.ravel(),
                bins=bins,
                title=f"Histogram: {label}",
                save_path=os.path.join(hist_root, f"{label}_hist.png")
            )

            for plane_name, sl in get_middle_slices(band_arr).items():
                plot_single_slice(
                    normalize_for_display(sl),
                    title=f"{label} - {plane_name}",
                    save_path=os.path.join(slice_root, f"{label}_{plane_name}.png")
                )

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="3D Medical Volume Wavelet Analysis")

    parser.add_argument("--input", type=str, required=True, help="Path to 3D input volume (.npy, .nii, .nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save analysis results")
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet name, e.g. db4, sym4, haar")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition level")
    parser.add_argument("--mode", type=str, default="symmetric", help="Boundary mode for wavelet transform")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for histograms / entropy")
    parser.add_argument("--clip_min", type=float, default=None, help="Optional minimum intensity clip")
    parser.add_argument("--clip_max", type=float, default=None, help="Optional maximum intensity clip")
    parser.add_argument("--save_coeffs", action="store_true", help="Save coefficient arrays as .npy")
    parser.add_argument("--use_abs_for_hist", action="store_true", help="Use abs(coefficients) for histogram plotting/entropy")

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    volume, metadata = load_volume(args.input)

    original_min = float(np.min(volume))
    original_max = float(np.max(volume))
    original_mean = float(np.mean(volume))
    original_std = float(np.std(volume))

    if args.clip_min is not None or args.clip_max is not None:
        clip_min = args.clip_min if args.clip_min is not None else float(np.min(volume))
        clip_max = args.clip_max if args.clip_max is not None else float(np.max(volume))
        volume = np.clip(volume, clip_min, clip_max)

    if args.level < 1:
        raise ValueError("Wavelet level must be >= 1")

    max_level = pywt.dwtn_max_level(volume.shape, wavelet=args.wavelet)
    if args.level > max_level:
        warnings.warn(
            f"Requested level={args.level} exceeds max allowed level={max_level} "
            f"for shape={volume.shape} and wavelet={args.wavelet}. "
            f"Using level={max_level} instead."
        )
        args.level = max_level

    # Save original volume representative slices
    original_slice_dir = os.path.join(args.output_dir, "original_slices")
    ensure_dir(original_slice_dir)
    for plane_name, sl in get_middle_slices(volume).items():
        plot_single_slice(
            normalize_for_display(sl),
            title=f"Original Volume - {plane_name}",
            save_path=os.path.join(original_slice_dir, f"original_{plane_name}.png")
        )

    # Run analysis
    df_stats = run_wavelet_analysis(
        volume=volume,
        wavelet_name=args.wavelet,
        level=args.level,
        mode=args.mode,
        bins=args.bins,
        output_dir=args.output_dir,
        save_coeffs=args.save_coeffs,
        use_abs_for_hist=args.use_abs_for_hist
    )

    # Save stats CSV
    csv_path = os.path.join(args.output_dir, "wavelet_subband_statistics.csv")
    df_stats.to_csv(csv_path, index=False)

    # Save energy plot
    plot_energy_bar(
        df_stats[["level", "subband", "energy"]].copy(),
        save_path=os.path.join(args.output_dir, "subband_energy_barplot.png")
    )

    # Save summary metadata
    summary = {
        "input_path": args.input,
        "output_dir": args.output_dir,
        "wavelet": args.wavelet,
        "level_used": args.level,
        "mode": args.mode,
        "bins": args.bins,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "save_coeffs": bool(args.save_coeffs),
        "use_abs_for_hist": bool(args.use_abs_for_hist),
        "volume_shape": tuple(int(x) for x in volume.shape),
        "volume_dtype": str(volume.dtype),
        "original_min": original_min,
        "original_max": original_max,
        "original_mean": original_mean,
        "original_std": original_std,
        "postprocess_min": float(np.min(volume)),
        "postprocess_max": float(np.max(volume)),
        "postprocess_mean": float(np.mean(volume)),
        "postprocess_std": float(np.std(volume)),
        "spacing": metadata.get("spacing", None),
        "source_type": metadata.get("source_type", None),
    }
    save_json(summary, os.path.join(args.output_dir, "analysis_summary.json"))

    # Simple console report
    print("\nAnalysis completed successfully.")
    print(f"Input volume shape: {volume.shape}")
    print(f"Wavelet: {args.wavelet}")
    print(f"Level: {args.level}")
    print(f"Mode: {args.mode}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Statistics CSV: {csv_path}")


if __name__ == "__main__":
    main()