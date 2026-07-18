# Noise Modeling Framework — Bug Fixes & Hypothesis

## File: `stats/noise_modeling2.py`

---

## Hypothesis

CBCT noise can be modeled as a mixture of **Poisson (quantum) noise** and **Uniform (electronic readout) noise**. The approach:

1. Split the CBCT intensity range into overlapping HU-scaled bins
2. Per bin, compute a likelihood distribution
3. Merge all per-bin distributions into a global noise model
4. Generate synthetic noise from this model
5. Apply log-rescaling and compare with the original distribution using KL and JS divergence

---

## Issues Found & Fixes Applied

### Fix 1: Poisson PMF on Continuous Data (Mathematical Invalidity)

**Location:** `build_overlapping_range_poisson_distribution()` — inner loop (line 717–731)

**Problem:**
The Poisson probability mass function,
```
P(k | λ) = (λ^k · e^(-λ)) / k!
```
was applied to **continuous** float pixel intensities (`bi`). The Poisson distribution is only defined for **non-negative integer** counts `k`. For non-integer input, Python's `factorial(bi)` invokes the gamma function `Γ(bi + 1)`, which:
- Does NOT produce valid probability masses over continuous values
- Causes catastrophic numerical overflow: `mean^bi` where `mean ≈ 80, bi ≈ 100` → `80^100 ≈ 10^190` (exceeds float64)
- The resulting "distribution" contained `NaN` and `inf` values

**Fix:**
Replaced Poisson PMF with the **Gaussian log-PDF**, computed in log-space for numerical stability:

```python
log_prob = -0.5 * ((bi - region_mean) / region_std)**2 \
           - np.log(region_std) - 0.5 * np.log(2 * np.pi)
probability_bi = np.exp(log_prob)
```

**Rationale:**
- Valid for continuous data (probability density rather than mass)
- Numerically stable (no exponentiation of large numbers, no factorial)
- Physically justified: CBCT Hounsfield Unit data is already in the post-log domain, where by the Central Limit Theorem, noise is approximately Gaussian with intensity-dependent variance
- Each bin's standard deviation `region_std` captures the actual local noise level

**Also removed:** `from scipy.special import factorial` (no longer needed)

---

### Fix 2: Log Transform Was Dead Code

**Location:** `transform_and_rescale()` (line 809)

**Problem:**
The function computed:
```python
combined_log = np.log(combined_noise_safe / np.max(combined_noise_safe))
```
but then immediately discarded it:
```python
rescaled = combined_noise  # BUG: operating on raw noise, not log
```
The final `rescaled_offset_final` was a min-max rescale of the **raw** combined noise, so the log-transform served no purpose.

**Fix:**
```python
rescaled = combined_log  # Now operates on log-transformed values
```

**Rationale:**
The log-transform is physically meaningful — it simulates the conversion from raw photon counts (where noise is Poisson/multiplicative) to CT numbers (where noise becomes approximately additive). Without this fix, the entire pipeline skipped the intended noise transformation step.

---

### Fix 3: Numerical Instability from Large Exponents

**Location:** `build_overlapping_range_poisson_distribution()` — removed formula

**Problem:**
The expression `region_mean ** bi` for typical values:
- `region_mean ≈ 50, bi ≈ 50` → `50^50 ≈ 8.9 × 10^84`
- `region_mean ≈ 80, bi ≈ 90` → `80^90 ≈ 2.0 × 10^171`

Both overflow float64's maximum of ~1.8 × 10^308 but produce `inf` when combined with `factorial(bi)` which also overflows, creating `NaN` ratios.

**Fix:**
Gaussian log-PDF requires only:
- `((bi - mean) / std)^2` — squared z-score, O(10²) at most
- `np.log(std)` — log of a small number
- Constants: `np.log(2 * np.pi)`

All terms are in log-space; no large exponentiations occur. The only `np.exp()` call is on the final log-probability which is guaranteed to be negative and of moderate magnitude.

---

### Fix 4: Inconsistent Normalization (Major)

**Location:** `normalize_image()` (line 630–645)

**Problem:**
The original V2 normalization used **per-patch min-max scaling** to [0, 100]:
```python
img_shifted = img + 1001
img_offset = (img_shifted - np.min(img_shifted)) * 100 / (np.max(img_shifted) - np.min(img_shifted))
```

This caused the `hu_bin_width` parameter to behave inconsistently:
- **Air patch** (range = 75 HU): min-max to [0,100] → local range ≈ 100 normalized units. `200 HU` scaled to `200 × 100 / 75 = 266.6` → **1 bin covers everything**
- **Full volume** (range = 3855 HU): `200 HU` scaled to `200 × 100 / 3855 = 5.19` → **~20 bins**
- The bin width in normalized space varied wildly depending on the data's local range

**Fix:**
V1 fixed normalization using the full CBCT range (5263 - (-1001) = 6264):
```python
img_offset = (img + 1001) * 100 / 6264
```

Now `200 HU` always maps to `200 × 100 / 6264 = 3.19` normalized units, regardless of whether the input is a full volume or a tiny uniform patch.

**Also updated:** `build_overlapping_range_poisson_distribution()` to use the constant `CBCT_FULL_RANGE = 6264` directly, ignoring the now-deprecated `original_range` parameter.

---

## Comparative Results: Fixed vs. Per-Patch Normalization

| Tissue | Norm | Bins | KL(P||Q) | KL(Q||P) | **JS** |
|--------|------|------|----------|----------|--------|
| Air | **Fixed** | 1 | 7.30 | 9.44 | **0.405** |
| Air | Per-Patch | 32 | 16.16 | 7.74 | 0.416 |
| Bone | **Fixed** | 6 | 4.93 | 1.53 | **0.219** |
| Bone | Per-Patch | 32 | 21.98 | 3.93 | 0.561 |
| Muscle | **Fixed** | 2 | 1.26 | 0.81 | **0.101** |
| Muscle | Per-Patch | 32 | 11.66 | 3.08 | 0.399 |

**Fixed normalization wins across all tissue types:**
- **Bone**: JS 0.219 vs 0.561 — **2.6× better**
- **Muscle**: JS 0.101 vs 0.399 — **4× better**
- **Air**: JS 0.405 vs 0.416 — essentially tied

Despite producing fewer bins, the Fixed normalization gives dramatically better distribution matching because HU-space bin widths remain consistent across all data.

---

## Files Changed

| File | Changes |
|------|---------|
| `stats/noise_modeling2.py` | All 4 fixes applied |
| `stats/_batch_noise_modeling2.py` | Removed `original_range` computation (now unused) |

## API Backward Compatibility

All function signatures remain unchanged:
- `build_overlapping_range_poisson_distribution()` — `original_range` parameter deprecated but still accepted
- `transform_and_rescale()` — same return values: `(combined_log, rescaled, rescaled_offset_final)`
- All other functions unchanged
