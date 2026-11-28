"""Shared forecasting evaluation — real-only, no synthetic in val/test.

Metrics are organized by their statistical unit:

- **Hourly cell** (primary): one (forecast origin o, lead time h) pair.
  Used for overall MAE/RMSE, lead-time breakdown.
- **Threshold event** (per cell): threshold_exceeded[y_o+h >= tau].
  Used for precision/recall/FPR/FAR at a train-only threshold.
- **Independent storm event**: contiguous block of hourly cells from one
  meteorologically-defined storm. Used for peak amplitude / timing error
  and paired bootstrap CI. Must NOT be treated as overlapping windows.

All thresholds q95/q99 come from train rows only. NaN handling: a cell where
either y or yhat is NaN contributes 0 to the denominator of cell-level metrics;
event-level metrics require that at least the peak cell is finite.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ==========================================================================
# Cell-level scalar metrics
# ==========================================================================


def _finite_mask(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(y, float)) & np.isfinite(np.asarray(yhat, float))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error over finite hourly cells."""
    m = _finite_mask(y, yhat)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y, float)[m] - np.asarray(yhat, float)[m])))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error over finite hourly cells."""
    m = _finite_mask(y, yhat)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(y, float)[m] - np.asarray(yhat, float)[m]) ** 2)))


def tail_mae(y: np.ndarray, yhat: np.ndarray, train_threshold: float) -> float:
    """MAE restricted to cells where y >= train_threshold (train-only)."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = _finite_mask(y, yhat) & (y >= train_threshold)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m])))


def tail_rmse(y: np.ndarray, yhat: np.ndarray, train_threshold: float) -> float:
    """RMSE restricted to cells where y >= train_threshold (train-only)."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = _finite_mask(y, yhat) & (y >= train_threshold)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


# ==========================================================================
# Threshold event metrics — explicit denominator semantics
# ==========================================================================


def threshold_metrics(y: np.ndarray, yhat: np.ndarray,
                      threshold: float) -> dict[str, float]:
    """Precision/recall/FPR/FAR at a train-only threshold.

    Definitions:
        TP = cells where y >= tau AND yhat >= tau (hit)
        FP = cells where y < tau AND yhat >= tau (false alarm)
        FN = cells where y >= tau AND yhat < tau (miss)
        TN = cells where y < tau AND yhat < tau (correct negative)

        Precision = TP / (TP + FP)           — what fraction of alarmed cells are real
        Recall    = TP / (TP + FN)           — what fraction of real extremes are alarmed
        F1        = 2 * P * R / (P + R)      — harmonic mean of precision and recall
        FPR       = FP / (FP + TN)            — false positive rate (specificity complement)
        FAR       = FP / (TP + FP)            — false alarm ratio (precision complement)

    Only finite cells contribute. All ratios return NaN when denominator is zero.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = _finite_mask(y, yhat)
    n_finite = int(m.sum())
    if n_finite == 0:
        return {k: float("nan") for k in [
            "precision", "recall", "f1", "fpr", "far",
            "tp", "fp", "fn", "tn", "n_finite", "threshold"]}

    pred_evt = (yhat >= threshold) & m
    true_evt = (y >= threshold) & m

    tp = float((pred_evt & true_evt).sum())
    fp = float((pred_evt & ~true_evt & m).sum())
    fn = float((~pred_evt & true_evt & m).sum())
    tn = float((~pred_evt & ~true_evt & m).sum())

    denom_p = tp + fp
    denom_r = tp + fn
    precision = tp / denom_p if denom_p > 0 else float("nan")
    recall = tp / denom_r if denom_r > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (
        np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
    ) else float("nan")
    denom_fpr = fp + tn
    fpr = fp / denom_fpr if denom_fpr > 0 else float("nan")
    far = fp / denom_p if denom_p > 0 else float("nan")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "far": far,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_finite": float(n_finite), "threshold": float(threshold),
    }


# ==========================================================================
# Event-level metrics (require event grouping)
# ==========================================================================


def peak_magnitude_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """|max(y) - max(yhat)| over finite cells."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if not np.isfinite(y).any() or not np.isfinite(yhat).any():
        return float("nan")
    return float(abs(np.nanmax(y) - np.nanmax(yhat)))


def peak_timing_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """|argmax(y_finite) - argmax(yhat_finite)| in steps.

    Finds the index of the finite-max element in each vector. If either vector
    is empty of finite values, returns NaN.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if not y.size or not yhat.size:
        return float("nan")
    y_finite = np.isfinite(y); yh_finite = np.isfinite(yhat)
    if not y_finite.any() or not yh_finite.any():
        return float("nan")
    # Use first occurrence if max ties
    i_y = int(np.argmax(np.where(y_finite, y, -np.inf)))
    i_hat = int(np.argmax(np.where(yh_finite, yhat, -np.inf)))
    return float(abs(i_y - i_hat))


# ==========================================================================
# Structural breakdowns
# ==========================================================================


def lead_time_metrics(y: np.ndarray, yhat: np.ndarray,
                      horizon: int | None = None) -> dict[str, np.ndarray]:
    """Compute MAE and RMSE per lead time.

    Args:
        y: 2D array of shape (n_forecasts, horizon) OR 1D flattened array.
        yhat: same shape as y.
        horizon: If None and y is 2D, inferred from shape. Required if y is 1D.

    Returns:
        dict with keys 'mae' and 'rmse', each an array of length horizon.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.ndim == 1:
        if horizon is None:
            raise ValueError("horizon required for 1D arrays")
        y = y.reshape(-1, horizon)
        yhat = yhat.reshape(-1, horizon)
    elif horizon is not None and y.shape[1] != horizon:
        raise ValueError(f"horizon {horizon} != y.shape[1] {y.shape[1]}")

    H = y.shape[1]
    mae_per_h = np.full(H, np.nan)
    rmse_per_h = np.full(H, np.nan)
    for h in range(H):
        mae_per_h[h] = mae(y[:, h], yhat[:, h])
        rmse_per_h[h] = rmse(y[:, h], yhat[:, h])
    return {"mae": mae_per_h, "rmse": rmse_per_h, "horizon": H}


def station_metrics(y: np.ndarray, yhat: np.ndarray,
                    station_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-station MAE and RMSE. station_ids must match axis 0 of 2D arrays."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.ndim != 2:
        raise ValueError("station_metrics requires 2D arrays: (stations × horizon or cells)")
    if len(station_ids) != y.shape[0]:
        raise ValueError(f"station_ids len {len(station_ids)} != y.shape[0] {y.shape[0]}")
    result: dict[str, dict[str, float]] = {}
    for i, sid in enumerate(station_ids):
        result[sid] = {
            "mae": mae(y[i], yhat[i]),
            "rmse": rmse(y[i], yhat[i]),
            "n_cells": float(int(_finite_mask(y[i], yhat[i]).sum())),
        }
    return result


# ==========================================================================
# Bootstrap CI — paired, event-unit
# ==========================================================================


def bootstrap_ci(values: Sequence[float], n_boot: int = 200,
                 alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """95% bootstrap CI for the mean of `values`.

    Inputs should already be at the independent-event unit level before
    this function is called (e.g., per-storm MAE, not per-window MAE).
    Overlapping-window values are NOT independent samples and would
    underestimate variance.

    Handles zero-event folds gracefully (returns NaN).
    """
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 3:
        return {
            "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "n_events": float(arr.size), "n_total_input": float(len(values)),
        }
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
             for _ in range(n_boot)]
    boots.sort()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "mean": float(np.mean(arr)), "lo": lo, "hi": hi,
        "n_events": float(arr.size), "n_total_input": float(len(values)),
    }


def paired_bootstrap_diff(values_a: Sequence[float], values_b: Sequence[float],
                          n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
                          ) -> dict[str, float]:
    """Paired bootstrap CI for mean(A - B).

    Both inputs must already be at the independent-event unit with a 1:1 pairing
    (e.g., for each storm, method-A MAE and method-B MAE). This preserves the
    event-specific variation that paired tests exploit.
    """
    a = np.array([v for v in values_a if np.isfinite(v)], dtype=float)
    b = np.array([v for v in values_b if np.isfinite(v)], dtype=float)
    n = min(a.size, b.size)
    if n < 3:
        return {"diff_mean": float("nan"), "diff_lo": float("nan"),
                "diff_hi": float("nan"), "n_pairs": float(n)}
    a = a[:n]; b = b[:n]
    diffs = a - b
    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append(float(np.mean(diffs[idx])))
    boot_diffs.sort()
    lo = float(np.quantile(boot_diffs, alpha / 2))
    hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
    return {"diff_mean": float(np.mean(diffs)), "diff_lo": lo, "diff_hi": hi,
            "n_pairs": float(n)}
