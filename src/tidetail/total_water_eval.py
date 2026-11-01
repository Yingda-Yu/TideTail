"""Total-water evaluation for TideTail.

Reconstructs predicted total water level from:
    pred_total_water = predicted_residual + future_astronomical_tide

Evaluates on the standard tide datum (MLLW), consistent across all variables.
Supports event-level metrics and paired bootstrap confidence intervals.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def reconstruct_total_water(residual: np.ndarray, tide: np.ndarray) -> np.ndarray:
    """predicted_total_water = residual + future_tide (both in meters, MLLW).

    Args:
        residual:  shape (N, H) — predicted or observed residuals
        tide:      shape (N, H) — future astronomical tide for each forecast horizon
    Returns:
        shape (N, H) — total water level
    """
    return residual + tide


def train_only_water_thresholds(total_water_train: np.ndarray) -> dict[str, float]:
    """Compute train-only q95/q99 thresholds on TOTAL WATER level.

    These are used for water-level Tail-MAE masking.
    """
    abs_water = np.abs(total_water_train)
    pos_water = total_water_train[total_water_train > 0]

    def _safe_percentile(arr: np.ndarray, q: float, fallback: float = np.nan) -> float:
        if len(arr) == 0:
            return fallback
        return float(np.percentile(arr, q))

    return {
        "water_q95_abs": _safe_percentile(abs_water, 95),
        "water_q99_abs": _safe_percentile(abs_water, 99),
        "water_q95_pos": _safe_percentile(pos_water, 95, fallback=_safe_percentile(abs_water, 95)),
        "water_q99_pos": _safe_percentile(pos_water, 99, fallback=_safe_percentile(abs_water, 99)),
    }


def _masked_tail_mae(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """MAE on masked subset. Returns NaN/undefined when n=0."""
    n = int(mask.sum())
    if n == 0:
        return {
            "tail_mae": float("nan"),
            "tail_rmse": float("nan"),
            "n": 0,
            "defined": False,
        }
    err = pred[mask] - tgt[mask]
    return {
        "tail_mae": float(np.mean(np.abs(err))),
        "tail_rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": n,
        "defined": True,
    }


def evaluate_total_water(
    pred_residual: np.ndarray,
    obs_residual: np.ndarray,
    future_tide: np.ndarray,
    train_water_q95_pos: float | None = None,
    train_water_q99_pos: float | None = None,
    train_water_q95_abs: float | None = None,
    train_water_q99_abs: float | None = None,
) -> dict[str, Any]:
    """Total-water evaluation metrics.

    Reconstructs predicted and observed total water, then computes:
      - Overall total-water MAE / RMSE
      - Train-only total-water q95/q99 Tail-MAE (on observed total water extremes)
      - Positive high-water peak error (per window, mean)
      - Peak timing error (per window, mean abs diff in steps)

    When no extreme targets exist at a given threshold, returns NaN + n=0.
    """
    pred_water = pred_residual + future_tide
    obs_water = obs_residual + future_tide

    # --- Overall total-water metrics ---
    err = pred_water - obs_water
    overall = {
        "tw_mae": float(np.mean(np.abs(err))),
        "tw_rmse": float(np.sqrt(np.mean(err ** 2))),
        "tw_bias": float(np.mean(err)),
    }

    # --- Total-water tail metrics (on TRAIN-ONLY thresholds, observed targets) ---
    tail = {}

    if train_water_q95_pos is not None and train_water_q99_pos is not None:
        high_pos_mask = obs_water >= train_water_q95_pos
        q95_pos = _masked_tail_mae(pred_water, obs_water, high_pos_mask)
        q95_pos["threshold"] = train_water_q95_pos
        q95_pos["description"] = f"total_water >= train_q95_pos ({train_water_q95_pos:.3f}m)"
        tail["tw_tail_mae_q95_pos"] = q95_pos

        high_pos_mask_99 = obs_water >= train_water_q99_pos
        q99_pos = _masked_tail_mae(pred_water, obs_water, high_pos_mask_99)
        q99_pos["threshold"] = train_water_q99_pos
        q99_pos["description"] = f"total_water >= train_q99_pos ({train_water_q99_pos:.3f}m)"
        tail["tw_tail_mae_q99_pos"] = q99_pos

    if train_water_q95_abs is not None and train_water_q99_abs is not None:
        high_abs_mask = np.abs(obs_water) >= train_water_q95_abs
        q95_abs = _masked_tail_mae(pred_water, obs_water, high_abs_mask)
        q95_abs["threshold"] = train_water_q95_abs
        q95_abs["description"] = f"|total_water| >= train_q95_abs ({train_water_q95_abs:.3f}m)"
        tail["tw_tail_mae_q95_abs"] = q95_abs

        high_abs_mask_99 = np.abs(obs_water) >= train_water_q99_abs
        q99_abs = _masked_tail_mae(pred_water, obs_water, high_abs_mask_99)
        q99_abs["threshold"] = train_water_q99_abs
        q99_abs["description"] = f"|total_water| >= train_q99_abs ({train_water_q99_abs:.3f}m)"
        tail["tw_tail_mae_q99_abs"] = q99_abs

    # --- Peak metrics (per window) ---
    peak_idx_obs = np.argmax(obs_water, axis=1)  # positive high-water peak
    peak_idx_pred = np.argmax(pred_water, axis=1)

    peak_err_per_window = np.array([
        pred_water[i, peak_idx_obs[i]] - obs_water[i, peak_idx_obs[i]]
        for i in range(len(obs_water))
    ])
    peak_timing_err = np.abs(peak_idx_pred - peak_idx_obs).astype(float)

    peaks = {
        "tw_peak_mae": float(np.mean(np.abs(peak_err_per_window))),
        "tw_peak_rmse": float(np.sqrt(np.mean(peak_err_per_window ** 2))),
        "tw_peak_bias": float(np.mean(peak_err_per_window)),
        "tw_peak_timing_mae_steps": float(np.mean(peak_timing_err)),
        "tw_peak_timing_median_steps": float(np.median(peak_timing_err)),
    }

    return {
        **overall,
        "total_water_tail": tail,
        "total_water_peaks": peaks,
    }


# ---------------------------------------------------------------------------
# Event-level aggregation + paired bootstrap CI
# ---------------------------------------------------------------------------

def event_level_metrics(
    metrics_per_window: list[dict[str, Any]],
    event_ids: list[str],
) -> dict[str, Any]:
    """Aggregate per-window metrics to per-event means.

    Each event's contribution is the mean of its constituent windows.
    Events provide the statistical unit for significance testing.
    """
    from collections import defaultdict

    # First attach event_id to each window metric
    event_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for met, eid in zip(metrics_per_window, event_ids):
        event_groups[eid].append(met)

    event_means: dict[str, dict[str, float]] = {}
    all_keys = set()
    for met in metrics_per_window:
        all_keys.update(met.keys())

    for eid, window_metrics in event_groups.items():
        em = {}
        for key in all_keys:
            vals = [wm[key] for wm in window_metrics if key in wm]
            vals_clean = [v for v in vals if not np.isnan(v)]
            if vals_clean:
                em[key] = float(np.mean(vals_clean))
            else:
                em[key] = float("nan")
        event_means[eid] = em

    return {
        "n_events": len(event_groups),
        "event_ids": sorted(event_groups.keys()),
        "event_means": event_means,
    }


def paired_bootstrap_ci(
    pred_water_a: np.ndarray,
    pred_water_b: np.ndarray,
    obs_water: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap CI for metric(A) - metric(B).

    Returns {estimate, ci_lo, ci_hi, boot_values}.
    """
    rng = np.random.default_rng(seed)
    n = len(obs_water)

    boot_a = np.zeros(n_boot)
    boot_b = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_a[b] = metric_fn(pred_water_a[idx], obs_water[idx])
        boot_b[b] = metric_fn(pred_water_b[idx], obs_water[idx])

    diff = boot_a - boot_b
    est = metric_fn(pred_water_a, obs_water) - metric_fn(pred_water_b, obs_water)
    ci_lo, ci_hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))

    return {
        "estimate": float(est),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "boot_estimate": float(np.mean(diff)),
        "n_boot": n_boot,
        "positive_better": ci_lo > 0,
        "negative_better": ci_hi < 0,
        "significant": ci_lo > 0 or ci_hi < 0,
    }
