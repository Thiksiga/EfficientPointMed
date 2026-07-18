# """
# Noise Component Extraction and Distribution Matching Framework
# -------------------------------------------------------------
#
# Purpose:
#     - Load full non-uniform CBCT image
#     - Normalize image intensity
#     - Split full intensity distribution into overlapping ranges
#     - Build Poisson likelihood per intensity range
#     - Merge all Poisson distributions
#     - Generate synthetic Poisson + Uniform noise
#     - Rescale and compare distributions using KL / JS divergence
# """
#
# # ============================================================
# # IMPORTS
# # ============================================================
#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import factorial
#
#
# # ============================================================
# # DATA LOADING & PREPROCESSING
# # ============================================================
#
# def load_image(path):
#     filename = os.path.basename(path)
#     print(f"Loaded file: {filename}")
#     img = np.load(path)
#     return img, filename
#
#
# def normalize_image(img):
#     """
#     Normalize original CBCT range approximately from [-1001, 5263]
#     into scaled positive range.
#     """
#     img_offset = img + 1001
#     img_offset = (img_offset*100)/(np.max(img_offset)-np.min(img_offset))
#     print(f"Normalized image shape: {img_offset.shape}")
#     print(
#         f"Normalized range -> min: {np.min(img_offset):.2f}, "
#         f"max: {np.max(img_offset):.2f}, "
#         f"mean: {np.mean(img_offset):.2f}"
#     )
#     return img_offset
#
#
# # ============================================================
# # OVERLAPPING INTENSITY-RANGE POISSON DISTRIBUTION
# # ============================================================
#
# # def build_overlapping_range_poisson_distribution(
# #     img_offset,
# #     bin_width=100,
# #     overlap_ratio=0.25
# # ):
# #     """
# #     Splits the full image intensity range into overlapping intensity bins.
# #
# #     For each overlapping intensity range:
# #         - collect pixels within that range
# #         - compute mean of that range
# #         - build Poisson likelihood using original formula:
# #               P(bi | mean) = (mean ** bi * exp(-mean)) / factorial(bi)
# #
# #     Then:
# #         - add all range-wise probability distributions
# #         - normalize the merged distribution
# #
# #     Overlap:
# #         bin_width = 100
# #         overlap_ratio = 0.25
# #         step = 75
# #         ranges:
# #             [0,100], [75,175], [150,250], ...
# #     """
# #
# #     l_b = np.min(img_offset)
# #     u_b = np.max(img_offset)
# #     global_mean = np.mean(img_offset)
# #
# #     overlap = bin_width * overlap_ratio
# #     step = bin_width - overlap
# #
# #     print("\n====== Overlapping Poisson Distribution Construction ======")
# #     print(f"Image intensity range: {l_b:.2f} to {u_b:.2f}")
# #     print(f"Global mean: {global_mean:.2f}")
# #     print(f"Bin width: {bin_width}")
# #     print(f"Overlap ratio: {overlap_ratio}")
# #     print(f"Overlap size: {overlap}")
# #     print(f"Step size: {step}")
# #
# #     merged_distribution = {}
# #
# #     start = l_b
# #     region_count = 0
# #
# #     while start < u_b:
# #         end = start + bin_width
# #
# #         region_mask = (img_offset >= start) & (img_offset <= end)
# #         region_pixels = img_offset[region_mask]
# #
# #         if region_pixels.size == 0:
# #             start += step
# #             continue
# #
# #         region_mean = np.mean(region_pixels)
# #
# #         print(
# #             f"Region {region_count:03d}: "
# #             f"[{start:.2f}, {end:.2f}] | "
# #             f"voxels: {region_pixels.size} | "
# #             f"mean: {region_mean:.4f}"
# #         )
# #
# #         # Keep your original Poisson calculation formula unchanged
# #         for bi in region_pixels:
# #             probability_bi = (
# #                 (region_mean ** bi) *
# #                 np.exp(-region_mean)
# #             ) / factorial(bi)
# #
# #             if bi in merged_distribution:
# #                 merged_distribution[bi] += probability_bi
# #             else:
# #                 merged_distribution[bi] = probability_bi
# #
# #         region_count += 1
# #         start += step
# #
# #     distribution_sorted = {
# #         k: merged_distribution[k]
# #         for k in sorted(merged_distribution.keys())
# #     }
# #
# #     total_probability = sum(distribution_sorted.values())
# #
# #     if total_probability == 0:
# #         raise ValueError("Merged Poisson distribution has zero total probability.")
# #
# #     distribution_sorted = {
# #         k: v / total_probability
# #         for k, v in distribution_sorted.items()
# #     }
# #
# #     print(f"Total overlapping regions processed: {region_count}")
# #     print(f"Final merged distribution size: {len(distribution_sorted)}")
# #
# #     return distribution_sorted, global_mean, l_b, u_b
#
# def build_overlapping_range_poisson_distribution(
#     img_offset,
#     hu_bin_width=100,
#     overlap_ratio=0.25,
#     original_range=6264,
#     normalized_range=100
# ):
#     """
#     Splits normalized image intensity range into overlapping bins.
#
#     hu_bin_width:
#         Bin width in original HU/intensity scale.
#         Example: 100 HU.
#
#     Since img_offset is normalized using:
#         img_offset = ((img + 1001) * 100 / 6264)
#
#     100 HU becomes:
#         100 * 100 / 6264 = 1.596 normalized units.
#     """
#
#     # Convert HU bin width to normalized scale
#     bin_width = hu_bin_width * normalized_range / original_range
#
#     l_b = np.min(img_offset)
#     u_b = np.max(img_offset)
#     global_mean = np.mean(img_offset)
#
#     overlap = bin_width * overlap_ratio
#     step = bin_width - overlap
#
#     print("\n====== Overlapping Poisson Distribution Construction ======")
#     print(f"Image intensity range (normalized): {l_b:.4f} to {u_b:.4f}")
#     print(f"Global mean (normalized): {global_mean:.4f}")
#     print(f"HU bin width: {hu_bin_width}")
#     print(f"Scaled bin width: {bin_width:.4f}")
#     print(f"Overlap ratio: {overlap_ratio}")
#     print(f"Scaled overlap size: {overlap:.4f}")
#     print(f"Scaled step size: {step:.4f}")
#
#     merged_distribution = {}
#
#     start = l_b
#     region_count = 0
#
#     while start < u_b:
#         end = start + bin_width
#
#         region_mask = (img_offset >= start) & (img_offset <= end)
#         region_pixels = img_offset[region_mask]
#
#         if region_pixels.size == 0:
#             start += step
#             continue
#
#         region_mean = np.mean(region_pixels)
#
#         print(
#             f"Region {region_count:03d}: "
#             f"[{start:.4f}, {end:.4f}] | "
#             f"voxels: {region_pixels.size} | "
#             f"mean: {region_mean:.4f}"
#         )
#
#         # Keep your original Poisson formula unchanged
#         for bi in region_pixels:
#             probability_bi = (
#                 (region_mean ** bi) *
#                 np.exp(-region_mean)
#             ) / factorial(bi)
#
#             if bi in merged_distribution:
#                 merged_distribution[bi] += probability_bi
#             else:
#                 merged_distribution[bi] = probability_bi
#
#         region_count += 1
#         start += step
#
#     distribution_sorted = {
#         k: merged_distribution[k]
#         for k in sorted(merged_distribution.keys())
#     }
#
#     total_probability = sum(distribution_sorted.values())
#
#     if total_probability == 0:
#         raise ValueError("Merged Poisson distribution has zero total probability.")
#
#     distribution_sorted = {
#         k: v / total_probability
#         for k, v in distribution_sorted.items()
#     }
#
#     print(f"Total overlapping regions processed: {region_count}")
#     print(f"Final merged distribution size: {len(distribution_sorted)}")
#
#     return distribution_sorted, global_mean, l_b, u_b
#
#
# # ============================================================
# # NOISE GENERATION
# # ============================================================
#
# def generate_custom_poisson_noise(distribution_sorted, shape):
#     keys = np.array(list(distribution_sorted.keys()))
#     values = np.array(list(distribution_sorted.values()))
#
#     values_normalized = values / values.sum()
#
#     noise_flat = np.random.choice(
#         keys,
#         size=np.prod(shape),
#         p=values_normalized
#     )
#
#     noise = noise_flat.reshape(shape)
#     return noise
#
#
# def generate_random_noise(shape, l_b, u_b, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#
#     noise = np.random.uniform(
#         low=l_b,
#         high=u_b,
#         size=shape
#     )
#
#     return noise
#
#
# # ============================================================
# # LOG TRANSFORMATION & RESCALING
# # ============================================================
#
# def transform_and_rescale(combined_noise, u_b, l_b):
#     eps = 1e-12
#
#     combined_noise_safe = np.clip(combined_noise, eps, None)
#     combined_log = np.log(combined_noise_safe / np.max(combined_noise_safe))
#
#     print(
#         f"combined_log -> min: {np.min(combined_log):.4f}, "
#         f"max: {np.max(combined_log):.4f}, "
#         f"mean: {np.mean(combined_log):.4f}, "
#         f"std: {np.std(combined_log):.4f}"
#     )
#
#     rescaled = combined_noise
#
#     rescaled_offset = rescaled - np.min(rescaled)
#
#     denominator = np.max(rescaled_offset) - np.min(rescaled_offset)
#
#     if denominator == 0:
#         raise ValueError("Cannot rescale: combined noise has constant value.")
#
#     rescaled_offset_final = (
#         rescaled_offset *
#         ((u_b - l_b) / denominator)
#     ) + l_b
#     # rescaled_offset_final = (rescaled_offset*(6264/100))
#
#     return combined_log, rescaled, rescaled_offset_final
#
#
# # ============================================================
# # DIVERGENCE UTILITIES
# # ============================================================
#
# def compute_pdf(data, bins=100, data_range=None, eps=1e-12):
#     hist, bin_edges = np.histogram(
#         data,
#         bins=bins,
#         range=data_range,
#         density=False
#     )
#
#     pdf = hist.astype(np.float64)
#     pdf /= np.sum(pdf)
#
#     pdf = np.clip(pdf, eps, None)
#     pdf /= np.sum(pdf)
#
#     return pdf, bin_edges
#
#
# def kl_divergence(P, Q):
#     return np.sum(P * np.log(P / Q))
#
#
# def js_divergence(P, Q):
#     M = 0.5 * (P + Q)
#     return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)
#
#
# # ============================================================
# # VISUALIZATION
# # ============================================================
#
# def plot_noise_statistics(
#     keys,
#     values,
#     poisson_noise,
#     uniform_noise,
#     combined_noise,
#     l_b,
#     u_b
# ):
#     fig, axes = plt.subplots(1, 4, figsize=(22, 5))
#
#     axes[0].bar(
#         keys,
#         values,
#         width=1,
#         color="steelblue",
#         edgecolor="none"
#     )
#     axes[0].set_title("Merged Overlapping Poisson Likelihood")
#     axes[0].set_xlabel("Pixel Intensity (k)")
#     axes[0].set_ylabel("Probability")
#
#     axes[1].hist(
#         poisson_noise.flatten(),
#         bins=100,
#         color="steelblue",
#         edgecolor="none"
#     )
#     axes[1].set_title("Generated Poisson Noise")
#
#     axes[2].hist(
#         uniform_noise.flatten(),
#         bins=100,
#         color="tomato",
#         edgecolor="none"
#     )
#     axes[2].set_title(f"Uniform Noise [{l_b:.2f}, {u_b:.2f}]")
#
#     axes[3].hist(
#         combined_noise.flatten(),
#         bins=100,
#         color="mediumseagreen",
#         edgecolor="none"
#     )
#     axes[3].set_title("Combined Noise Model")
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_pdf_comparison(P, Q, bin_edges, filename):
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(bin_centers, P, label="Original PDF")
#     plt.plot(bin_centers, Q, label="Generated / Rescaled PDF")
#
#     plt.xlabel("Intensity")
#     plt.ylabel("Probability")
#     plt.title(f"PDF Comparison - {filename}")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#
# # ============================================================
# # MAIN EXECUTION
# # ============================================================
#
# def main():
#     # --------------------------------------------------------
#     # Parameters
#     # --------------------------------------------------------
#
#     path = r"D:\Mphil\Dataset\P50\data.npy"
#
#     noise_shape = (5, 10, 10)
#
#     # Control factors
#     a = 1.0       # Poisson noise contribution
#     b = 0.01      # Uniform noise contribution
#
#     # Overlapping distribution parameters
#     bin_width = 100
#     overlap_ratio = 0.25
#
#     # --------------------------------------------------------
#     # Load and normalize image
#     # --------------------------------------------------------
#
#     img, filename = load_image(path)
#     original_range = img.max() - img.min()
#     img_offset = normalize_image(img)
#
#     # --------------------------------------------------------
#     # Build overlapping range-wise Poisson distribution
#     # --------------------------------------------------------
#
#     distribution_sorted, mean, l_b, u_b = (
#         build_overlapping_range_poisson_distribution(
#             img_offset,
#             hu_bin_width=bin_width,
#             # original_range=original_range,
#             overlap_ratio=overlap_ratio
#         )
#     )
#
#     keys = list(distribution_sorted.keys())
#     values = list(distribution_sorted.values())
#
#     # --------------------------------------------------------
#     # Noise generation
#     # --------------------------------------------------------
#
#     poisson_noise = generate_custom_poisson_noise(
#         distribution_sorted,
#         shape=noise_shape
#     )
#
#     uniform_noise = generate_random_noise(
#         shape=noise_shape,
#         l_b=l_b,
#         u_b=u_b,
#         seed=25
#     )
#
#     combined_noise = (a * poisson_noise) + (b * uniform_noise)
#
#     combined_log, rescaled, rescaled_offset_final = (
#         transform_and_rescale(
#             combined_noise,
#             u_b,
#             l_b
#         )
#     )
#
#     # --------------------------------------------------------
#     # Statistics
#     # --------------------------------------------------------
#
#     print("\n====== Noise Statistics ======")
#
#     print(
#         f"Poisson noise  -> min: {np.min(poisson_noise):.2f}, "
#         f"max: {np.max(poisson_noise):.2f}, "
#         f"mean: {np.mean(poisson_noise):.2f}, "
#         f"std: {np.std(poisson_noise):.2f}"
#     )
#
#     print(
#         f"Uniform noise  -> min: {np.min(uniform_noise):.2f}, "
#         f"max: {np.max(uniform_noise):.2f}, "
#         f"mean: {np.mean(uniform_noise):.2f}, "
#         f"std: {np.std(uniform_noise):.2f}"
#     )
#
#     print(
#         f"Combined noise -> min: {np.min(combined_noise):.2f}, "
#         f"max: {np.max(combined_noise):.2f}, "
#         f"mean: {np.mean(combined_noise):.2f}, "
#         f"std: {np.std(combined_noise):.2f}"
#     )
#
#     print(
#         f"Rescaled noise -> min: {np.min(rescaled_offset_final):.2f}, "
#         f"max: {np.max(rescaled_offset_final):.2f}, "
#         f"mean: {np.mean(rescaled_offset_final):.2f}, "
#         f"std: {np.std(rescaled_offset_final):.2f}"
#     )
#
#     # --------------------------------------------------------
#     # Visualization
#     # --------------------------------------------------------
#
#     plot_noise_statistics(
#         keys,
#         values,
#         poisson_noise,
#         uniform_noise,
#         combined_noise,
#         l_b,
#         u_b
#     )
#
#     # --------------------------------------------------------
#     # Divergence analysis
#     # --------------------------------------------------------
#
#     img_flat = img_offset.flatten()
#     rescaled_flat = rescaled_offset_final.flatten()
#
#     common_min = min(img_flat.min(), rescaled_flat.min())
#     common_max = max(img_flat.max(), rescaled_flat.max())
#
#     P, bin_edges = compute_pdf(
#         img_flat,
#         bins=100,
#         data_range=(common_min, common_max)
#     )
#
#     Q, _ = compute_pdf(
#         rescaled_flat,
#         bins=100,
#         data_range=(common_min, common_max)
#     )
#
#     kl_PQ = kl_divergence(P, Q)
#     kl_QP = kl_divergence(Q, P)
#     js = js_divergence(P, Q)
#
#     print("\n====== Divergence Analysis ======")
#     print("KL(img || generated): %.6f" % kl_PQ)
#     print("KL(generated || img): %.6f" % kl_QP)
#     print("JS divergence: %.6f" % js)
#
#     plot_pdf_comparison(
#         P,
#         Q,
#         bin_edges,
#         filename
#     )
#
#
# # ============================================================
# # ENTRY POINT
# # ============================================================
#
# if __name__ == "__main__":
#     main()


"""
Noise Component Extraction and Distribution Matching Framework
-------------------------------------------------------------

Purpose:
    - Load full non-uniform CBCT image
    - Normalize image intensity
    - Split full intensity distribution into overlapping HU-equivalent ranges
    - Build Poisson likelihood per intensity range
    - Merge all Poisson distributions
    - Generate synthetic Poisson + Uniform noise
    - Rescale and compare distributions using KL / JS divergence
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SEEDING
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_image(path):
    filename = os.path.basename(path)
    print(f"Loaded file: {filename}")
    img = np.load(path)
    return img, filename


def normalize_image(img):
    """
    Fixed normalization: shift by +1001, scale to [0, 100] using the
    full CBCT range (6264 HU) so that hu_bin_width maps consistently
    across all patches and volumes.
    """
    img_offset = (img + 1001) * 100 / 6264

    print(f"Normalized image shape: {img_offset.shape}")
    print(
        f"Normalized range -> min: {np.min(img_offset):.2f}, "
        f"max: {np.max(img_offset):.2f}, "
        f"mean: {np.mean(img_offset):.2f}"
    )

    return img_offset


# ============================================================
# OVERLAPPING POISSON DISTRIBUTION
# ============================================================

CBCT_FULL_RANGE = 6264  # fixed: 5263 - (-1001)


def build_overlapping_range_poisson_distribution(
    img_offset,
    hu_bin_width=100,
    overlap_ratio=0.25,
    original_range=None,
    normalized_range=100
):
    """
    Build merged Poisson distribution over overlapping intensity ranges.

    hu_bin_width is given in original HU/intensity scale.
    It is converted to normalized image scale using the fixed
    CBCT_FULL_RANGE (6264) since normalize_image now uses fixed scaling.

    Example:
        100 HU -> 100 * 100 / 6264 normalized units.
    """

    bin_width = hu_bin_width * normalized_range / CBCT_FULL_RANGE

    l_b = np.min(img_offset)
    u_b = np.max(img_offset)
    global_mean = np.mean(img_offset)

    overlap = bin_width * overlap_ratio
    step = bin_width - overlap

    print("\n====== Overlapping Poisson Distribution Construction ======")
    print(f"Image intensity range normalized: {l_b:.4f} to {u_b:.4f}")
    print(f"Global mean normalized: {global_mean:.4f}")
    print(f"CBCT full range: {CBCT_FULL_RANGE}")
    print(f"HU bin width: {hu_bin_width}")
    print(f"Scaled bin width: {bin_width:.4f}")
    print(f"Overlap ratio: {overlap_ratio}")
    print(f"Scaled overlap size: {overlap:.4f}")
    print(f"Scaled step size: {step:.4f}")

    merged_distribution = {}

    start = l_b
    region_count = 0

    while start < u_b:
        end = start + bin_width

        region_mask = (img_offset >= start) & (img_offset <= end)
        region_pixels = img_offset[region_mask]

        if region_pixels.size == 0:
            start += step
            continue

        region_mean = np.mean(region_pixels)

        print(
            f"Region {region_count:03d}: "
            f"[{start:.4f}, {end:.4f}] | "
            f"voxels: {region_pixels.size} | "
            f"mean: {region_mean:.4f}"
        )

        region_std = np.std(region_pixels)
        if region_std < 1e-8:
            region_std = 1e-8

        for bi in region_pixels:
            log_prob = (
                -0.5 * ((bi - region_mean) / region_std) ** 2
                - np.log(region_std)
                - 0.5 * np.log(2 * np.pi)
            )
            probability_bi = np.exp(log_prob)

            if bi in merged_distribution:
                merged_distribution[bi] += probability_bi
            else:
                merged_distribution[bi] = probability_bi

        region_count += 1
        start += step

    distribution_sorted = {
        k: merged_distribution[k]
        for k in sorted(merged_distribution.keys())
    }

    total_probability = sum(distribution_sorted.values())

    if total_probability == 0:
        raise ValueError("Merged Poisson distribution has zero total probability.")

    distribution_sorted = {
        k: v / total_probability
        for k, v in distribution_sorted.items()
    }

    print(f"Total overlapping regions processed: {region_count}")
    print(f"Final merged distribution size: {len(distribution_sorted)}")

    return distribution_sorted, global_mean, l_b, u_b


# ============================================================
# NOISE GENERATION
# ============================================================

def generate_custom_poisson_noise(distribution_sorted, shape, seed=42):
    rng = np.random.default_rng(seed)

    keys = np.array(list(distribution_sorted.keys()))
    values = np.array(list(distribution_sorted.values()))

    values_normalized = values / values.sum()

    noise_flat = rng.choice(
        keys,
        size=np.prod(shape),
        p=values_normalized
    )

    noise = noise_flat.reshape(shape)

    return noise


def generate_random_noise(shape, l_b, u_b, seed=42):
    rng = np.random.default_rng(seed)

    noise = rng.uniform(
        low=l_b,
        high=u_b,
        size=shape
    )

    return noise


# ============================================================
# RESCALING
# ============================================================

def transform_and_rescale(combined_noise, u_b, l_b):
    eps = 1e-12

    combined_noise_safe = np.clip(combined_noise, eps, None)
    combined_log = np.log(combined_noise_safe / np.max(combined_noise_safe))

    print(
        f"combined_log -> min: {np.min(combined_log):.4f}, "
        f"max: {np.max(combined_log):.4f}, "
        f"mean: {np.mean(combined_log):.4f}, "
        f"std: {np.std(combined_log):.4f}"
    )

    rescaled = combined_log

    rescaled_offset = rescaled - np.min(rescaled)
    denominator = np.max(rescaled_offset) - np.min(rescaled_offset)

    if denominator < 1e-12:
        raise ValueError("Cannot rescale: combined noise has constant value.")

    rescaled_offset_final = (
        rescaled_offset *
        ((u_b - l_b) / denominator)
    ) + l_b

    return combined_log, combined_noise, rescaled_offset_final


# ============================================================
# DIVERGENCE UTILITIES
# ============================================================

def compute_pdf(data, bins=100, data_range=None, eps=1e-12):
    hist, bin_edges = np.histogram(
        data,
        bins=bins,
        range=data_range,
        density=False
    )

    pdf = hist.astype(np.float64)
    pdf /= np.sum(pdf)

    pdf = np.clip(pdf, eps, None)
    pdf /= np.sum(pdf)

    return pdf, bin_edges


def kl_divergence(P, Q):
    return np.sum(P * np.log(P / Q))


def js_divergence(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_noise_statistics(
    keys,
    values,
    poisson_noise,
    uniform_noise,
    combined_noise,
    l_b,
    u_b
):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].bar(
        keys,
        values,
        width=1,
        color="steelblue",
        edgecolor="none"
    )
    axes[0].set_title("Merged Overlapping Poisson Likelihood")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Probability")

    axes[1].hist(
        poisson_noise.flatten(),
        bins=100,
        color="steelblue",
        edgecolor="none"
    )
    axes[1].set_title("Generated Poisson Noise")

    axes[2].hist(
        uniform_noise.flatten(),
        bins=100,
        color="tomato",
        edgecolor="none"
    )
    axes[2].set_title(f"Uniform Noise [{l_b:.2f}, {u_b:.2f}]")

    axes[3].hist(
        combined_noise.flatten(),
        bins=100,
        color="mediumseagreen",
        edgecolor="none"
    )
    axes[3].set_title("Combined Noise Model")

    plt.tight_layout()
    plt.show()


def plot_pdf_comparison(P, Q, bin_edges, filename):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, P, label="Original PDF")
    plt.plot(bin_centers, Q, label="Generated / Rescaled PDF")

    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title(f"PDF Comparison - {filename}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    set_seed(42)

    path = r"D:\Mphil\Dataset\P50\data.npy"
    # path = r"D:\Mphil\Dataset_overview\P50_stats\bony_115_3d.npy"
    # path = r"D:\Mphil\Dataset_overview\P20_stats\air_225_3d.npy"
    # path = r"D:\Mphil\Dataset_overview\P20_stats\bony_115_3d.npy"
    # path = r"D:\Mphil\Dataset_overview\P20_stats\soft_tissue_200_3d.npy"

    a = 1.0
    b = 0.01

    hu_bin_width = 200
    overlap_ratio = 0.05

    img, filename = load_image(path)

    img_offset = normalize_image(img)

    # Use full image shape to avoid small-sample distribution offset
    noise_shape = img_offset.shape

    distribution_sorted, mean, l_b, u_b = (
        build_overlapping_range_poisson_distribution(
            img_offset,
            hu_bin_width=hu_bin_width,
            overlap_ratio=overlap_ratio,
        )
    )

    keys = list(distribution_sorted.keys())
    values = list(distribution_sorted.values())

    poisson_noise = generate_custom_poisson_noise(
        distribution_sorted,
        shape=noise_shape,
        seed=42
    )

    uniform_noise = generate_random_noise(
        shape=noise_shape,
        l_b=l_b,
        u_b=u_b,
        seed=25
    )

    combined_noise = (a * poisson_noise) + (b * uniform_noise)

    combined_log, rescaled, rescaled_offset_final = (
        transform_and_rescale(
            combined_noise,
            u_b,
            l_b
        )
    )

    print("\n====== Noise Statistics ======")

    print(
        f"Poisson noise  -> min: {np.min(poisson_noise):.2f}, "
        f"max: {np.max(poisson_noise):.2f}, "
        f"mean: {np.mean(poisson_noise):.2f}, "
        f"std: {np.std(poisson_noise):.2f}"
    )

    print(
        f"Uniform noise  -> min: {np.min(uniform_noise):.2f}, "
        f"max: {np.max(uniform_noise):.2f}, "
        f"mean: {np.mean(uniform_noise):.2f}, "
        f"std: {np.std(uniform_noise):.2f}"
    )

    print(
        f"Combined noise -> min: {np.min(combined_noise):.2f}, "
        f"max: {np.max(combined_noise):.2f}, "
        f"mean: {np.mean(combined_noise):.2f}, "
        f"std: {np.std(combined_noise):.2f}"
    )

    print(
        f"Rescaled noise -> min: {np.min(rescaled_offset_final):.2f}, "
        f"max: {np.max(rescaled_offset_final):.2f}, "
        f"mean: {np.mean(rescaled_offset_final):.2f}, "
        f"std: {np.std(rescaled_offset_final):.2f}"
    )

    plot_noise_statistics(
        keys,
        values,
        poisson_noise,
        uniform_noise,
        combined_noise,
        l_b,
        u_b
    )

    img_flat = img_offset.flatten()
    rescaled_flat = rescaled_offset_final.flatten()

    common_min = min(img_flat.min(), rescaled_flat.min())
    common_max = max(img_flat.max(), rescaled_flat.max())

    P, bin_edges = compute_pdf(
        img_flat,
        bins=100,
        data_range=(common_min, common_max)
    )

    Q, _ = compute_pdf(
        rescaled_flat,
        bins=100,
        data_range=(common_min, common_max)
    )

    kl_PQ = kl_divergence(P, Q)
    kl_QP = kl_divergence(Q, P)
    js = js_divergence(P, Q)

    print("\n====== Divergence Analysis ======")
    print("KL(img || generated): %.6f" % kl_PQ)
    print("KL(generated || img): %.6f" % kl_QP)
    print("JS divergence: %.6f" % js)

    plot_pdf_comparison(
        P,
        Q,
        bin_edges,
        filename
    )


if __name__ == "__main__":
    main()