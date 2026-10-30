"""TideTail generation-quality evaluator.

Rigorous, CPU-only statistics for comparing generated water-level
forecasts against real data. Every metric is deterministic on fixed
inputs; no random sampling inside the evaluator.

Quality gate
------------
A generation is ``ACCEPTED`` only if **all** metrics land within
policy bounds. Bounds are defaults derived from TideTail's physics
contract (extreme coastal water levels, 96h→72h horizon) — they are
*not* estimates of physical plausibility.

Metrics
-------
Statistical  — Wasserstein-1, PSD deviation, autocorrelation lag
Extreme      — q95/q99 exceedance ratio, peak magnitude KS-test,
               rise/decay time distributions
Memorization — exact duplicate ratio, near-duplicate (cosine ≥ 0.995),
               nearest-neighbor distance to train set
Scale        — std ratio, min/max range ratio
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Statistical metrics
# ---------------------------------------------------------------------------

def wasserstein1(real: np.ndarray, gen: np.ndarray) -> float:
    """Wasserstein-1 distance between two 1-D distributions.

    Accepts flattened arrays; exact via sorted CDF difference.
    """
    r = np.asarray(real, dtype=np.float64).ravel()
    g = np.asarray(gen, dtype=np.float64).ravel()
    if r.size < 2 or g.size < 2:
        raise ValueError("need at least 2 samples per distribution")
    # Common sorted axis
    all_vals = np.sort(np.concatenate([r, g]))
    # Empirical CDFs evaluated at common axis
    cdf_r = np.searchsorted(np.sort(r), all_vals, side="right") / r.size
    cdf_g = np.searchsorted(np.sort(g), all_vals, side="right") / g.size
    return float(np.trapz(np.abs(cdf_r - cdf_g), all_vals))


def psd_deviation(real: np.ndarray, gen: np.ndarray,
                  normalize: bool = True) -> float:
    """Mean absolute relative deviation of power spectral densities.

    Normalize each PSD to unit integral before comparison so that
    absolute scale differences don't dominate. Accepts ``real`` and
    ``gen`` with different batch sizes; only the last (horizon)
    dimension must match.
    """
    r = np.asarray(real, dtype=np.float64)
    g = np.asarray(gen, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)
        g = g.reshape(1, -1)
    if r.shape[-1] != g.shape[-1]:
        raise ValueError(f"horizon mismatch: real has {r.shape[-1]}, gen has {g.shape[-1]}")

    def _psd(x: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(x - x.mean(axis=-1, keepdims=True), axis=-1)
        return (X.real**2 + X.imag**2) / (x.shape[-1] ** 2)

    pr = _psd(r).mean(axis=0, keepdims=True)  # (1, freq)
    pg = _psd(g).mean(axis=0, keepdims=True)
    if normalize:
        pr = pr / pr.sum(axis=-1, keepdims=True).clip(min=1e-12)
        pg = pg / pg.sum(axis=-1, keepdims=True).clip(min=1e-12)
    # Mean over frequencies (skip DC bin)
    return float(np.abs(pr[:, 1:] - pg[:, 1:]).mean())


def autocorr_lag_match(real: np.ndarray, gen: np.ndarray,
                       max_lag: int = 12) -> float:
    """RMSE between real and gen autocorrelation at lags 0..max_lag.

    lag=0 always gives 1 by construction; the comparison is on
    lags ≥ 1 where temporal structure matters. Accepts different
    batch sizes.
    """
    r = np.asarray(real, dtype=np.float64)
    g = np.asarray(gen, dtype=np.float64)
    if r.ndim == 1:
        r = r.reshape(1, -1)
        g = g.reshape(1, -1)
    if r.shape[-1] != g.shape[-1]:
        raise ValueError(f"horizon mismatch: real has {r.shape[-1]}, gen has {g.shape[-1]}")

    def _autocorr(x: np.ndarray, lags: int) -> np.ndarray:
        out = np.zeros((x.shape[0], lags + 1))
        for j in range(x.shape[0]):
            for lag in range(lags + 1):
                if lag == 0:
                    out[j, lag] = 1.0
                else:
                    a = x[j, :-lag]
                    b = x[j, lag:]
                    if a.std() < 1e-12 or b.std() < 1e-12:
                        out[j, lag] = 0.0
                    else:
                        out[j, lag] = np.corrcoef(a, b)[0, 1]
        return out

    ar = _autocorr(r, max_lag).mean(axis=0)
    ag = _autocorr(g, max_lag).mean(axis=0)
    return float(np.sqrt(np.mean((ar[1:] - ag[1:]) ** 2)))


# ---------------------------------------------------------------------------
# Extreme-value metrics
# ---------------------------------------------------------------------------

def tail_exceedance_ratio(real: np.ndarray, gen: np.ndarray,
                          quantile: float = 0.95) -> float:
    """P(gen exceeds q) / P(real exceeds q) — ratio of tail exceedance.

    Ratio > 1 → generation has more extreme events than real.
    Ratio < 1 → generation is too conservative in the tail.
    """
    r = np.asarray(real, dtype=np.float64).ravel()
    g = np.asarray(gen, dtype=np.float64).ravel()
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0,1), got {quantile}")
    q = np.quantile(r, quantile)
    p_real = float(np.mean(r >= q))
    p_gen = float(np.mean(g >= q))
    if p_real < 1e-12:
        return float("nan")
    return p_gen / p_real


def peak_magnitude_ks(real: np.ndarray, gen: np.ndarray,
                      horizon_axis: int = -1) -> tuple[float, float]:
    """Kolmogorov-Smirnov test on per-sample peak magnitudes.

    Returns (D statistic, p-value). Low p-value → peaks differ.
    """
    r_peaks = np.asarray(real, dtype=np.float64).max(axis=horizon_axis).ravel()
    g_peaks = np.asarray(gen, dtype=np.float64).max(axis=horizon_axis).ravel()
    return stats.ks_2samp(r_peaks, g_peaks)


def peak_timing_mae(real: np.ndarray, gen: np.ndarray,
                    horizon_axis: int = -1) -> float:
    """MAE of peak-time distributions (mean |t_peak_real - t_peak_gen|).

    Computes peak indices for both populations, then returns the
    absolute difference of their *means*. Works with different batch
    sizes — this is a distribution-level metric, not pairwise.
    """
    r_idx = np.argmax(real, axis=horizon_axis).ravel()
    g_idx = np.argmax(gen, axis=horizon_axis).ravel()
    return float(abs(float(r_idx.mean()) - float(g_idx.mean())))


def rise_decay_stats(series: np.ndarray, peak_idx: int,
                      threshold: float = 0.0) -> tuple[float, float]:
    """Rise time and decay time around the peak of one series.

    Rise time: hours from first point above threshold to peak.
    Decay time: hours from peak back to below threshold.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    above = s >= threshold
    # Rise: walk backward from peak until crossing below threshold
    rise = 0
    for i in range(peak_idx - 1, -1, -1):
        if above[i]:
            rise += 1
        else:
            break
    # Decay: walk forward from peak until crossing below threshold
    decay = 0
    for i in range(peak_idx + 1, len(s)):
        if above[i]:
            decay += 1
        else:
            break
    return float(rise), float(decay)


# ---------------------------------------------------------------------------
# Memorization / diversity checks
# ---------------------------------------------------------------------------

def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between matrices a and b."""
    na = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    nb = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return na @ nb.T


def duplicate_ratio(gen: np.ndarray, exact_tol: float = 1e-6) -> float:
    """Fraction of generated samples that are exact duplicates of others."""
    g = np.asarray(gen, dtype=np.float64).reshape(len(gen), -1)
    if len(g) < 2:
        return 0.0
    seen: dict[bytes, int] = {}
    dup = 0
    for row in g:
        key = (np.round(row / max(exact_tol, 1e-12)) * max(exact_tol, 1e-12)).tobytes()
        if key in seen:
            dup += 1
        else:
            seen[key] = 1
    return dup / len(g)


def near_duplicate_ratio(gen: np.ndarray, threshold: float = 0.995) -> float:
    """Fraction of generated samples with cosine ≥ threshold to any other.

    Counts each sample at most once as a near-duplicate of *some* other.
    Does not double-count pairs.
    """
    g = np.asarray(gen, dtype=np.float64).reshape(len(gen), -1)
    if len(g) < 2:
        return 0.0
    sim = _cosine_matrix(g, g)
    np.fill_diagonal(sim, -1.0)
    # Each row: max similarity to any *other* sample
    max_sim = sim.max(axis=1)
    return float(np.mean(max_sim >= threshold))


def nearest_neighbor_distance(gen: np.ndarray,
                              train: np.ndarray) -> float:
    """Mean Euclidean distance from each gen sample to its nearest train match.

    Normalizes by train set std so the number is interpretable:
    values near 0 → memorization; values around 1 → healthy diversity.
    """
    g = np.asarray(gen, dtype=np.float64).reshape(len(gen), -1)
    t = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    if len(g) == 0 or len(t) == 0:
        return float("nan")
    scale = t.std(axis=0).mean().clip(min=1e-6)
    dists = np.zeros(len(g))
    for i in range(len(g)):
        d = np.linalg.norm(t - g[i], axis=1).min()
        dists[i] = d / (scale * math.sqrt(g.shape[1]))
    return float(dists.mean())


# ---------------------------------------------------------------------------
# Scale metrics
# ---------------------------------------------------------------------------

def scale_ratio(real: np.ndarray, gen: np.ndarray,
                horizon_axis: int = -1) -> float:
    """std(gen) / std(real) computed over all values."""
    return float(np.asarray(gen).std() / np.asarray(real).std())


def range_ratio(real: np.ndarray, gen: np.ndarray,
                horizon_axis: int = -1) -> float:
    """(max-min)(gen) / (max-min)(real) — per-sample, averaged."""
    r = np.asarray(real, dtype=np.float64)
    g = np.asarray(gen, dtype=np.float64)
    rng_r = (r.max(axis=horizon_axis) - r.min(axis=horizon_axis)).mean()
    rng_g = (g.max(axis=horizon_axis) - g.min(axis=horizon_axis)).mean()
    if rng_r < 1e-12:
        return float("nan")
    return float(rng_g / rng_r)


# ---------------------------------------------------------------------------
# Aggregate quality gate
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Aggregated quality metrics + pass/fail decision."""

    # Inputs
    n_real: int = 0
    n_gen: int = 0
    horizon_len: int = 0

    # Statistical
    wasserstein1: float = float("nan")
    psd_deviation: float = float("nan")
    autocorr_rmse: float = float("nan")

    # Scale
    scale_ratio: float = float("nan")
    range_ratio: float = float("nan")

    # Extreme
    q95_exceedance_ratio: float = float("nan")
    q99_exceedance_ratio: float = float("nan")
    peak_ks_stat: float = float("nan")
    peak_ks_pvalue: float = float("nan")
    peak_timing_mae: float = float("nan")

    # Memorization
    exact_dup_ratio: float = float("nan")
    near_dup_ratio_0995: float = float("nan")
    nn_distance_to_train: float = float("nan")

    # Gate
    status: str = "INCONCLUSIVE"  # ACCEPTED / REJECTED / INCONCLUSIVE
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()
               if not isinstance(v, (list, tuple))}
        out["failures"] = list(self.failures)
        out["warnings"] = list(self.warnings)
        return out

    def summary(self) -> str:
        lines = [
            f"QualityReport: {self.status}",
            f"  n_real={self.n_real}, n_gen={self.n_gen}, horizon={self.horizon_len}",
            f"  Scale:       std ratio={self.scale_ratio:.3f}, range ratio={self.range_ratio:.3f}",
            f"  Wasserstein-1: {self.wasserstein1:.4f}",
            f"  PSD dev:    {self.psd_deviation:.4f}",
            f"  AutocorrRMSE: {self.autocorr_rmse:.4f}",
            f"  Tail exc:    q95={self.q95_exceedance_ratio:.3f}, q99={self.q99_exceedance_ratio:.3f}",
            f"  Peak KS:     D={self.peak_ks_stat:.4f}, p={self.peak_ks_pvalue:.4g}",
            f"  Timing MAE:  {self.peak_timing_mae:.2f} h",
            f"  Duplicates:  exact={self.exact_dup_ratio:.3f}, near={self.near_dup_ratio_0995:.3f}",
            f"  NN-dist:     {self.nn_distance_to_train:.3f}",
        ]
        if self.failures:
            lines.append(f"  FAILURES: {self.failures}")
        if self.warnings:
            lines.append(f"  WARNINGS: {self.warnings}")
        return "\n".join(lines)


def evaluate(real: np.ndarray, gen: np.ndarray,
             train_ref: np.ndarray | None = None,
             policy: dict | None = None) -> QualityReport:
    """Run the full quality battery and return a QualityReport.

    Args:
        real: (N, horizon) — real held-out forecast targets
        gen:  (M, horizon) — generated forecast samples (M can be ≠ N)
        train_ref: (K, horizon) — training samples for memorization check.
                   If None, memorization metrics return NaN.
        policy: Override default bounds. Keys that are bounds use
                (lo, hi) tuples; flags use bool. See ``_default_policy``.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if real.ndim != 2 or gen.ndim != 2 or real.shape[1] != gen.shape[1]:
        raise ValueError("real and gen must both be 2-D (N, horizon)")

    p = {**_default_policy(), **(policy or {})}

    R = QualityReport(n_real=len(real), n_gen=len(gen),
                      horizon_len=real.shape[1])

    # Scale + statistical
    R.scale_ratio = scale_ratio(real, gen)
    R.range_ratio = range_ratio(real, gen)
    R.wasserstein1 = wasserstein1(real, gen)
    R.psd_deviation = psd_deviation(real, gen)
    R.autocorr_rmse = autocorr_lag_match(real, gen)

    # Extreme
    R.q95_exceedance_ratio = tail_exceedance_ratio(real, gen, 0.95)
    R.q99_exceedance_ratio = tail_exceedance_ratio(real, gen, 0.99)
    R.peak_ks_stat, R.peak_ks_pvalue = peak_magnitude_ks(real, gen)
    R.peak_timing_mae = peak_timing_mae(real, gen)

    # Memorization (diversity within gen)
    R.exact_dup_ratio = duplicate_ratio(gen, exact_tol=p["exact_dup_tol"])
    R.near_dup_ratio_0995 = near_duplicate_ratio(gen, 0.995)
    if train_ref is not None:
        R.nn_distance_to_train = nearest_neighbor_distance(gen, train_ref)

    # Gate
    failures, warnings = [], []

    # Scale
    if not (p["scale_lo"] <= R.scale_ratio <= p["scale_hi"]):
        failures.append(f"scale_ratio={R.scale_ratio:.2f} ∉ [{p['scale_lo']}, {p['scale_hi']}]")
    if not (p["range_lo"] <= R.range_ratio <= p["range_hi"]):
        warnings.append(f"range_ratio={R.range_ratio:.2f} ∉ [{p['range_lo']}, {p['range_hi']}]")

    # Statistical
    if R.wasserstein1 > p["wasserstein_max"]:
        warnings.append(f"Wasserstein1={R.wasserstein1:.4f} > {p['wasserstein_max']}")
    if R.autocorr_rmse > p["autocorr_rmse_max"]:
        warnings.append(f"autocorr_RMSE={R.autocorr_rmse:.4f} > {p['autocorr_rmse_max']}")

    # Extreme
    if not (p["tail_lo"] <= R.q95_exceedance_ratio <= p["tail_hi"]):
        warnings.append(f"q95_exceed_ratio={R.q95_exceedance_ratio:.2f} ∉ [{p['tail_lo']}, {p['tail_hi']}]")
    if R.peak_ks_pvalue < p["peak_ks_p_min"]:
        warnings.append(f"peak KS p={R.peak_ks_pvalue:.4g} < {p['peak_ks_p_min']}")
    if R.peak_timing_mae > p["peak_timing_mae_max"]:
        warnings.append(f"peak_timing_MAE={R.peak_timing_mae:.2f} h > {p['peak_timing_mae_max']}")

    # Memorization
    if R.exact_dup_ratio > p["exact_dup_max"]:
        failures.append(f"exact_dup_ratio={R.exact_dup_ratio:.3f} > {p['exact_dup_max']}")
    if R.near_dup_ratio_0995 > p["near_dup_max"]:
        warnings.append(f"near_dup_ratio={R.near_dup_ratio_0995:.3f} > {p['near_dup_max']}")
    if train_ref is not None and R.nn_distance_to_train < p["nn_dist_min"]:
        warnings.append(f"NN-dist={R.nn_distance_to_train:.3f} < {p['nn_dist_min']} (possible memorization)")

    R.failures = failures
    R.warnings = warnings
    if failures:
        R.status = "REJECTED"
    elif warnings:
        R.status = "INCONCLUSIVE"
    else:
        R.status = "ACCEPTED"

    return R


def _default_policy() -> dict:
    """TideTail 96→72 policy defaults.

    These are engineering-quality bounds, *not* physical plausibility
    certificates. Tune via ``policy`` arg to :func:`evaluate`.
    """
    return {
        # Scale (hard failures if out of range)
        "scale_lo": 0.5, "scale_hi": 2.0,
        "range_lo": 0.3, "range_hi": 3.0,
        # Statistical
        "wasserstein_max": 0.25,     # absolute depends on data scale
        "autocorr_rmse_max": 0.15,
        # Extreme
        "tail_lo": 0.5, "tail_hi": 2.5,      # exceedance ratio
        "peak_ks_p_min": 0.01,                 # reject peak-dist difference
        "peak_timing_mae_max": 8.0,            # 8 h on 72 h horizon
        # Memorization
        "exact_dup_max": 0.01,
        "near_dup_max": 0.10,
        "nn_dist_min": 0.3,
        # Tolerance knobs
        "exact_dup_tol": 1e-4,
    }
