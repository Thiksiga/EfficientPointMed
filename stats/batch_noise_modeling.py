"""
Batch Noise Modeling - Process all P10/P20 slices and save plots
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import factorial

sys.path.insert(0, str(Path(__file__).parent))

from noise_modeling import (
    load_image,
    normalize_image,
    build_custom_poisson_distribution,
    generate_custom_poisson_noise,
    generate_random_noise,
    transform_and_rescale,
    compute_pdf,
    kl_divergence,
    js_divergence,
)


def plot_noise_statistics_save(keys, values, poisson_noise, uniform_noise,
                               rescaled_offset_final, l_b, u_b, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].bar(keys, values, width=1, color='steelblue', edgecolor='none')
    axes[0].set_title("Custom Poisson Likelihood")
    axes[0].set_xlabel("Pixel Intensity (k)")
    axes[0].set_ylabel("Poisson Likelihood P(k|mu)")

    axes[1].hist(poisson_noise.flatten(), bins=100, color='steelblue', edgecolor='none')
    axes[1].set_title("Custom Poisson Noise")

    axes[2].hist(uniform_noise.flatten(), bins=100, color='tomato', edgecolor='none')
    axes[2].set_title(f"Uniform Random Noise [{l_b/10:.0f}, {u_b/10:.0f}]")

    axes[3].hist(rescaled_offset_final.flatten(), bins=100,
                 color='mediumseagreen', edgecolor='none')
    axes[3].set_title("Combined Noise model")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pdf_comparison_save(P, Q, bin_edges, filename, save_path):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, P, label="Original PDF")
    ax.plot(bin_centers, Q, label="Rescaled PDF")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Probability")
    ax.set_title(f"PDF Comparison (100 bins) {filename}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_slice(slice_path: Path, output_dir: Path):
    noise_shape = (5, 10, 10)
    a = 1
    b = 0.01

    img, filename = load_image(str(slice_path))
    img_offset = normalize_image(img)

    distribution_sorted, mean, l_b, u_b = build_custom_poisson_distribution(img_offset)

    keys = list(distribution_sorted.keys())
    values = list(distribution_sorted.values())

    poisson_noise = generate_custom_poisson_noise(distribution_sorted, shape=noise_shape)
    uniform_noise = generate_random_noise(shape=noise_shape, l_b=l_b, u_b=u_b, seed=25)

    combined_noise = (a * poisson_noise) + (b * uniform_noise)

    combined_log, rescaled, rescaled_offset_final = transform_and_rescale(combined_noise, u_b, l_b)

    noise_stats_path = output_dir / "noise_statistics.png"
    plot_noise_statistics_save(keys, values, poisson_noise, uniform_noise,
                                rescaled_offset_final, l_b, u_b, noise_stats_path)

    img_flat = img_offset.flatten()
    rescaled_flat = rescaled_offset_final.flatten()

    common_min = min(img_flat.min(), rescaled_flat.min())
    common_max = max(img_flat.max(), rescaled_flat.max())

    P, bin_edges = compute_pdf(img_flat, bins=100, data_range=(common_min, common_max))
    Q, _ = compute_pdf(rescaled_flat, bins=100, data_range=(common_min, common_max))

    kl_PQ = kl_divergence(P, Q)
    kl_QP = kl_divergence(Q, P)
    js = js_divergence(P, Q)

    pdf_comp_path = output_dir / "pdf_comparison.png"
    plot_pdf_comparison_save(P, Q, bin_edges, filename, pdf_comp_path)

    return kl_PQ, kl_QP, js


def discover_slices(slices_dir: Path):
    return sorted(slices_dir.glob("*.npy"))


def main():
    base = Path("./Results")
    plots_base = Path("./Results/Plots")

    for group in ["P10", "P20"]:
        slices_dir = base / group / "Slices"
        if not slices_dir.exists():
            print(f"Directory not found: {slices_dir}")
            continue

        cat_dirs = [d for d in slices_dir.iterdir() if d.is_dir()]
        if cat_dirs:
            for cat_dir in cat_dirs:
                slices = discover_slices(cat_dir)
                prefix = cat_dir.name.lower()
                print(f"\n{group}/{cat_dir.name}: {len(slices)} slices")
                for slice_path in slices:
                    slice_name = slice_path.stem.replace("air_", f"{prefix}_")
                    out_dir = plots_base / group / slice_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        kl_PQ, kl_QP, js = process_slice(slice_path, out_dir)
                        print(f"  [OK] {slice_name}: KL(img||rescaled)={kl_PQ:.6f}, KL(rescaled||img)={kl_QP:.6f}, JS={js:.6f}")
                    except Exception as e:
                        print(f"  [FAIL] {slice_name}: {e}")
        else:
            slices = discover_slices(slices_dir)
            print(f"\n{group}: {len(slices)} slices")
            for slice_path in slices:
                slice_name = slice_path.stem
                out_dir = plots_base / group / slice_name
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    kl_PQ, kl_QP, js = process_slice(slice_path, out_dir)
                    print(f"  [OK] {slice_name}: KL(img||rescaled)={kl_PQ:.6f}, KL(rescaled||img)={kl_QP:.6f}, JS={js:.6f}")
                except Exception as e:
                    print(f"  [FAIL] {slice_name}: {e}")


if __name__ == "__main__":
    main()