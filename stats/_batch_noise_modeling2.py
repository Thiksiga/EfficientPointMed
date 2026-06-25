"""
Temporary automation script for noise_modeling2.py — batch processes P10 slices.
Does NOT modify noise_modeling2.py.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from noise_modeling2 import (
    load_image,
    normalize_image,
    build_overlapping_range_poisson_distribution,
    generate_custom_poisson_noise,
    generate_random_noise,
    transform_and_rescale,
    compute_pdf,
    kl_divergence,
    js_divergence,
    set_seed,
)


def plot_noise_statistics_save(keys, values, poisson_noise, uniform_noise,
                               combined_noise, l_b, u_b, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].bar(keys, values, width=1, color='steelblue', edgecolor='none')
    axes[0].set_title("Merged Overlapping Poisson Likelihood")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Probability")

    axes[1].hist(poisson_noise.flatten(), bins=100, color='steelblue', edgecolor='none')
    axes[1].set_title("Generated Poisson Noise")

    axes[2].hist(uniform_noise.flatten(), bins=100, color='tomato', edgecolor='none')
    axes[2].set_title(f"Uniform Noise [{l_b:.2f}, {u_b:.2f}]")

    axes[3].hist(combined_noise.flatten(), bins=100,
                 color='mediumseagreen', edgecolor='none')
    axes[3].set_title("Combined Noise Model")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pdf_comparison_save(P, Q, bin_edges, filename, save_path):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, P, label="Original PDF")
    ax.plot(bin_centers, Q, label="Generated / Rescaled PDF")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Probability")
    ax.set_title(f"PDF Comparison - {filename}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_slice(slice_path: Path, output_dir: Path, a=1.0, b=0.01,
                  hu_bin_width=200, overlap_ratio=0.05):
    set_seed(42)

    img, filename = load_image(str(slice_path))

    original_range = np.max(img) - np.min(img)
    img_offset = normalize_image(img)

    noise_shape = img_offset.shape

    distribution_sorted, mean, l_b, u_b = (
        build_overlapping_range_poisson_distribution(
            img_offset,
            hu_bin_width=hu_bin_width,
            overlap_ratio=overlap_ratio,
            original_range=original_range
        )
    )

    keys = list(distribution_sorted.keys())
    values = list(distribution_sorted.values())

    poisson_noise = generate_custom_poisson_noise(
        distribution_sorted, shape=noise_shape, seed=42
    )

    uniform_noise = generate_random_noise(
        shape=noise_shape, l_b=l_b, u_b=u_b, seed=25
    )

    combined_noise = (a * poisson_noise) + (b * uniform_noise)

    combined_log, rescaled, rescaled_offset_final = transform_and_rescale(
        combined_noise, u_b, l_b
    )

    noise_stats_path = output_dir / "noise_statistics.png"
    plot_noise_statistics_save(keys, values, poisson_noise, uniform_noise,
                               combined_noise, l_b, u_b, noise_stats_path)

    img_flat = img_offset.flatten()
    rescaled_flat = rescaled_offset_final.flatten()

    common_min = min(img_flat.min(), rescaled_flat.min())
    common_max = max(img_flat.max(), rescaled_flat.max())

    P, bin_edges = compute_pdf(img_flat, bins=100,
                               data_range=(common_min, common_max))
    Q, _ = compute_pdf(rescaled_flat, bins=100,
                       data_range=(common_min, common_max))

    kl_PQ = kl_divergence(P, Q)
    kl_QP = kl_divergence(Q, P)
    js = js_divergence(P, Q)

    pdf_comp_path = output_dir / "pdf_comparison.png"
    plot_pdf_comparison_save(P, Q, bin_edges, filename, pdf_comp_path)

    noise_stats = {
        "poisson_min": np.min(poisson_noise), "poisson_max": np.max(poisson_noise),
        "poisson_mean": np.mean(poisson_noise), "poisson_std": np.std(poisson_noise),
        "uniform_min": np.min(uniform_noise), "uniform_max": np.max(uniform_noise),
        "uniform_mean": np.mean(uniform_noise), "uniform_std": np.std(uniform_noise),
        "combined_min": np.min(combined_noise), "combined_max": np.max(combined_noise),
        "combined_mean": np.mean(combined_noise), "combined_std": np.std(combined_noise),
        "rescaled_min": np.min(rescaled_offset_final), "rescaled_max": np.max(rescaled_offset_final),
        "rescaled_mean": np.mean(rescaled_offset_final), "rescaled_std": np.std(rescaled_offset_final),
        "kl_PQ": kl_PQ, "kl_QP": kl_QP, "js": js,
    }

    return noise_stats


def main():
    base = Path("./Results")
    group = "P10"
    slices_dir = base / group / "Slices"

    csv_header = (
        "slice,"
        "Poisson noise min,Poisson noise max,Poisson noise mean,Poisson noise std,"
        "Uniform noise min,Uniform noise max,Uniform noise mean,Uniform noise std,"
        "Combined noise min,Combined noise max,Combined noise mean,Combined noise std,"
        "Rescaled noise min,Rescaled noise max,Rescaled noise mean,Rescaled noise std,"
        "KL(img || generated),KL(generated || img),JS divergence"
    )

    cat_dirs = [d for d in slices_dir.iterdir() if d.is_dir()]
    targets_by_cat = {}
    if cat_dirs:
        for cat_dir in cat_dirs:
            prefix = cat_dir.name.lower()
            targets_by_cat[prefix] = sorted(cat_dir.glob("*.npy"))
    else:
        targets_by_cat[""] = sorted(slices_dir.glob("*.npy"))

    for prefix, targets in targets_by_cat.items():
        if prefix:
            csv_path = base / group / prefix.capitalize() / "noise_modeling_results.csv"
        else:
            csv_path = base / group / "noise_modeling_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        all_rows = [csv_header]

        for slice_path in targets:
            slice_name = slice_path.stem.replace("air_", f"{prefix}_") if prefix else slice_path.stem
            out_dir = Path("./Results/Plots") / group / slice_name
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                s = process_slice(slice_path, out_dir)
                row = (
                    f"{slice_name},"
                    f"{s['poisson_min']:.2f},{s['poisson_max']:.2f},{s['poisson_mean']:.2f},{s['poisson_std']:.2f},"
                    f"{s['uniform_min']:.2f},{s['uniform_max']:.2f},{s['uniform_mean']:.2f},{s['uniform_std']:.2f},"
                    f"{s['combined_min']:.2f},{s['combined_max']:.2f},{s['combined_mean']:.2f},{s['combined_std']:.2f},"
                    f"{s['rescaled_min']:.2f},{s['rescaled_max']:.2f},{s['rescaled_mean']:.2f},{s['rescaled_std']:.2f},"
                    f"{s['kl_PQ']:.6f},{s['kl_QP']:.6f},{s['js']:.6f}"
                )
                all_rows.append(row)
                print(f"  [OK] {slice_name}: KL(img||gen)={s['kl_PQ']:.6f}, KL(gen||img)={s['kl_QP']:.6f}, JS={s['js']:.6f}")
            except Exception as e:
                print(f"  [FAIL] {slice_name}: {e}")
                import traceback
                traceback.print_exc()

        csv_path.write_text("\n".join(all_rows), encoding="utf-8")
        print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
