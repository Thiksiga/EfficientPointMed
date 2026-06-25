"""
Noise Component Extraction and Distribution Matching Framework
---------------------------------------------------------------

Purpose:
    - Extract noise from uniform medical regions
    - Model custom Poisson likelihood
    - Generate synthetic noise (Poisson + Uniform)
    - Apply log-rescaling transformation
    - Compare distributions using KL and JS divergence
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import factorial


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_image(path):
    filename = os.path.basename(path)
    print(filename)
    img = np.load(path)
    return img, filename


def normalize_image(img):
    img_offset = (img + 1001) * 100 / 6264
    print(img_offset.shape)
    return img_offset


# ============================================================
# CUSTOM POISSON LIKELIHOOD CONSTRUCTION
# ============================================================

def build_custom_poisson_distribution(img_offset):
    mean = np.mean(img_offset)
    l_b = np.min(img_offset)
    u_b = np.max(img_offset)

    print(f"mu: {mean: .2f}, u_b: {u_b: .2f}, l_b: {l_b: .2f}")

    distribution = {}

    for slice in img_offset:
        for row in slice:
            for bi in row:
                probability_bi = ((mean ** bi) * np.exp(-mean)) / (factorial(bi))
                distribution[bi] = probability_bi

    distribution_sorted = {k: distribution[k] for k in sorted(distribution.keys())}

    return distribution_sorted, mean, l_b, u_b


# ============================================================
# NOISE GENERATION
# ============================================================

def generate_custom_poisson_noise(distribution_sorted, shape):
    keys = np.array(list(distribution_sorted.keys()))
    values = np.array(list(distribution_sorted.values()))

    values_normalized = values / values.sum()

    noise_flat = np.random.choice(keys, size=np.prod(shape), p=values_normalized)
    noise = noise_flat.reshape(shape)
    return noise


def generate_random_noise(shape, l_b, u_b, seed=None):
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.uniform(low=l_b, high=u_b, size=shape)
    return noise


# ============================================================
# LOG TRANSFORMATION & RESCALING
# ============================================================

def transform_and_rescale(combined_noise, u_b, l_b):

    combined_log = u_b - (np.log(combined_noise / np.max(combined_noise)))

    print(
        f"combined_log --> min: {np.min(combined_log)}, "
        f"max: {np.max(combined_log)}, "
        f"mean: {np.mean(combined_log)}, "
        f"std: {np.std(combined_log)}"
    )

    rescaled = (((u_b - combined_log) / (1)))
    rescaled_offset = (rescaled - np.min(rescaled))

    rescaled_offset_final = (
        rescaled_offset *
        ((u_b - l_b) / (np.max(rescaled_offset) - np.min(rescaled_offset)))
    ) + l_b

    return combined_log, rescaled, rescaled_offset_final


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

def plot_noise_statistics(keys, values,
                          poisson_noise,
                          uniform_noise,
                          rescaled_offset_final,
                          l_b, u_b):

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].bar(keys, values, width=1, color='steelblue', edgecolor='none')
    axes[0].set_title("Custom Poisson Likelihood")
    axes[0].set_xlabel("Pixel Intensity (k)")
    axes[0].set_ylabel("Poisson Likelihood P(k|μ)")

    axes[1].hist(poisson_noise.flatten(), bins=100, color='steelblue', edgecolor='none')
    axes[1].set_title("Custom Poisson Noise")

    axes[2].hist(uniform_noise.flatten(), bins=100, color='tomato', edgecolor='none')
    axes[2].set_title(f"Uniform Random Noise [{l_b/10:.0f}, {u_b/10:.0f}]")

    axes[3].hist(rescaled_offset_final.flatten(), bins=100,
                 color='mediumseagreen', edgecolor='none')
    axes[3].set_title("Combined Noise model")

    plt.tight_layout()
    plt.show()


def plot_pdf_comparison(P, Q, bin_edges, filename):

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, P, label="Original PDF")
    plt.plot(bin_centers, Q, label="Rescaled PDF")

    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title(f"PDF Comparison (100 bins) {filename}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    # Parameters
    path = r"./Results/P10/Slices/air_52_3d.npy"
    noise_shape = (5, 10, 10)
    a = 1      # Control factor of poisson noise
    b = 0.01    # Control factor of uniform noise

    img, filename = load_image(path)
    img_offset = normalize_image(img)

    distribution_sorted, mean, l_b, u_b = \
        build_custom_poisson_distribution(img_offset)

    keys = list(distribution_sorted.keys())
    values = list(distribution_sorted.values())

    # Noise generation
    poisson_noise = generate_custom_poisson_noise(
        distribution_sorted,
        shape=noise_shape
    )

    uniform_noise = generate_random_noise(
        shape=noise_shape,
        l_b=l_b,
        u_b=u_b,
        seed=25
    )

    combined_noise = (a*poisson_noise) + (b*uniform_noise)

    combined_log, rescaled, rescaled_offset_final = \
        transform_and_rescale(combined_noise, u_b, l_b)

    print(f"Poisson noise  -> min: {np.min(poisson_noise):.2f}, "
          f"max: {np.max(poisson_noise):.2f}, "
          f"mean: {np.mean(poisson_noise):.2f}")

    print(f"Uniform noise  -> min: {np.min(uniform_noise):.2f}, "
          f"max: {np.max(uniform_noise):.2f}, "
          f"mean: {np.mean(uniform_noise):.2f}")

    print(f"Combined noise -> min: {np.min(combined_noise):.2f}, "
          f"max: {np.max(combined_noise):.2f}, "
          f"mean: {np.mean(combined_noise):.2f}")

    # Plot noise distributions
    plot_noise_statistics(keys, values,
                          poisson_noise,
                          uniform_noise,
                          rescaled_offset_final,
                          l_b, u_b)

    # Divergence analysis
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

    print("KL(img || rescaled): %.6f" % kl_PQ)
    print("KL(rescaled || img): %.6f" % kl_QP)
    print("JS divergence: %.6f" % js)

    plot_pdf_comparison(P, Q, bin_edges, filename)


# ============================================================
if __name__ == "__main__":
    main()
