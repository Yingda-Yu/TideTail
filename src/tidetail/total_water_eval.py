"""Total-water evaluation for TideTail.

Reconstructs predicted total water level from:
    pred_total_water = predicted_residual + future_astronomical_tide

Evaluates on the standard tide datum (MLLW), consistent across all variables.

PHYSICAL UNITS AND DATUMS (per FORECAST_TASK_CONTRACT.md):
    - All water-level variables: metres above MLLW
    - FES2022 tide: metres above MLLW (deterministic astronomical component)
    - Residual (water_level_m): metres above MLLW (de-tided surge)
    - Datum is CONSISTENT across all three variables

Peak metrics distinguish two fundamentally different questions:
    a) Peak magnitude error:       how far off is each window's OWN highest point?
    b) Error at observed peak:     how wrong is prediction EXACTLY when observed peaks?
       (This is timing+height conflated — the pred could be right height at wrong time)

Paired bootstrap resamples INDEPENDENT STORM EVENT IDs, not overlapping forecast
windows. Events are the correct sampling unit for significance testing because
windows from the same storm are highly correlated.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants (datum / physical bounds)
# ---------------------------------------------------------------------------

DATUM = "MLLW"  # metres above local mean lower low water
PHYSICAL_WATER_BOUNDS_M = (-5.0, 20.0)  # sanity bounds for tide-surge combined


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def reconstruct_total_water(residual: np.ndarray, tide: np.ndarray) -> np.ndarray:
    """predicted_total_water = residual + future_tide (both metres, MLLW).

    Contract: residual IS the de-tided surge (NOT double-detided — see
    docs/audits/CRITICAL_TARGET_SEMANTICS_AUDIT.md). tide IS FES2022.
    """
    out = residual + tide
    # Sanity check — no NaN/Inf in reconstruction
    assert not np.any(np.isnan(out)), "NaN in total-water reconstruction"
    assert not np.any(np.isinf(out)), "Inf in total-water reconstruction"
    return out


def train_only_water_thresholds(total_water_train: np.ndarray) -> dict[str, float]:
    """Compute train-only q95/q99 thresholds on TOTAL WATER level.

    These thresholds apply to the OBSERVED total water and are the ONLY
    source of truth for extreme-event masking. Must NOT be recomputed per
    split or per seed — they are frozen from the train set once.

    Returns dict with keys:
        water_q95_pos, water_q99_pos  — positive (flooding) extremes
        water_q95_abs, water_q99_abs  — absolute extremes
    """
    # Validate input
    assert not np.any(np.isnan(total_water_train)), "NaN in train total water"
    assert len(total_water_train) > 100, (
        f"train total water too small ({len(total_water_train)} rows)"
    )

    abs_water = np.abs(total_water_train)
    pos_water = total_water_train[total_water_train > 0]

    def _safe_percentile(arr: np.ndarray, q: float, fallback: float = np.nan) -> float:
        if len(arr) == 0:
            return fallback
        return float(np.percentile(arr, q))

    result = {
        "water_q95_abs": _safe_percentile(abs_water, 95),
        "water_q99_abs": _safe_percentile(abs_water, 99),
    }
    # Positive thresholds — if no positive values, fall back to abs thresholds
    pos_fallback = result["water_q95_abs"]
    result["water_q95_pos"] = _safe_percentile(pos_water, 95, fallback=pos_fallback)
    pos_fallback_99 = result["water_q99_abs"]
    result["water_q99_pos"] = _safe_percentile(pos_water, 99, fallback=pos_fallback_99)
    return result


def _masked_tail_mae(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """MAE on masked subset. Returns NaN + defined=False when n=0.

    NEVER falls back to overall MAE when no extreme targets exist.
    """
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


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

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

    All units in metres above MLLW (see DATUM constant).
    """
    pred_water = pred_residual + future_tide
    obs_water = obs_residual + future_tide

    # --- Overall total-water metrics ---
    err = pred_water - obs_water
    overall = {
        "tw_mae": float(np.mean(np.abs(err))),
        "tw_rmse": float(np.sqrt(np.mean(err ** 2))),
        "tw_bias": float(np.mean(err)),
        "tw_corr": float(np.corrcoef(
            pred_water.reshape(-1), obs_water.reshape(-1)
        )[0, 1]),
    }

    # --- Total-water tail metrics (TRAIN-ONLY thresholds, observed targets) ---
    tail = {}

    if train_water_q95_pos is not None and train_water_q99_pos is not None:
        high_pos = obs_water >= train_water_q95_pos
        q95_pos = _masked_tail_mae(pred_water, obs_water, high_pos)
        q95_pos["threshold"] = train_water_q95_pos
        q95_pos["mask_definition"] = (
            f"observed_total_water >= train_q95_pos ({train_water_q95_pos:.3f}m)"
        )
        tail["tw_tail_mae_q95_pos"] = q95_pos

        high_pos_99 = obs_water >= train_water_q99_pos
        q99_pos = _masked_tail_mae(pred_water, obs_water, high_pos_99)
        q99_pos["threshold"] = train_water_q99_pos
        q99_pos["mask_definition"] = (
            f"observed_total_water >= train_q99_pos ({train_water_q99_pos:.3f}m)"
        )
        tail["tw_tail_mae_q99_pos"] = q99_pos

    if train_water_q95_abs is not None and train_water_q99_abs is not None:
        high_abs = np.abs(obs_water) >= train_water_q95_abs
        q95_abs = _masked_tail_mae(pred_water, obs_water, high_abs)
        q95_abs["threshold"] = train_water_q95_abs
        q95_abs["mask_definition"] = (
            f"|observed_total_water| >= train_q95_abs ({train_water_q95_abs:.3f}m)"
        )
        tail["tw_tail_mae_q95_abs"] = q95_abs

        high_abs_99 = np.abs(obs_water) >= train_water_q99_abs
        q99_abs = _masked_tail_mae(pred_water, obs_water, high_abs_99)
        q99_abs["threshold"] = train_water_q99_abs
        q99_abs["mask_definition"] = (
            f"|observed_total_water| >= train_q99_abs ({train_water_q99_abs:.3f}m)"
        )
        tail["tw_tail_mae_q99_abs"] = q99_abs

    # --- Peak metrics (THREE distinct concepts) ---
    # 1. Peak magnitude error: |argmax(pred) - argmax(obs)| — each window's OWN peak height
    # 2. Error at observed peak time: pred[argmax(obs)] - obs[argmax(obs)] — timing+height conflated
    # 3. Peak timing error: |argmax(pred) - argmax(obs)| — just timing
    peak_idx_obs = np.argmax(obs_water, axis=1)
    peak_idx_pred = np.argmax(pred_water, axis=1)

    # Peak MAGNITUDE error: compare each window's own peak
    obs_peak_heights = obs_water[np.arange(len(obs_water)), peak_idx_obs]
    pred_peak_heights = pred_water[np.arange(len(pred_water)), peak_idx_pred]
    peak_mag_err = pred_peak_heights - obs_peak_heights

    # Error AT observed peak time: how wrong is pred exactly when obs peaks?
    pred_at_obs_peak = pred_water[np.arange(len(obs_water)), peak_idx_obs]
    err_at_obs_peak = pred_at_obs_peak - obs_peak_heights

    peak_timing_err = np.abs(peak_idx_pred - peak_idx_obs).astype(float)

    peaks = {
        # 1. Peak magnitude error (metres, compares heights)
        "tw_peak_mag_mae": float(np.mean(np.abs(peak_mag_err))),
        "tw_peak_mag_rmse": float(np.sqrt(np.mean(peak_mag_err ** 2))),
        "tw_peak_mag_bias": float(np.mean(peak_mag_err)),
        # 2. Error specifically AT the observed peak time
        "tw_err_at_obs_peak_mae": float(np.mean(np.abs(err_at_obs_peak))),
        "tw_err_at_obs_peak_rmse": float(np.sqrt(np.mean(err_at_obs_peak ** 2))),
        "tw_err_at_obs_peak_bias": float(np.mean(err_at_obs_peak)),
        # 3. Peak timing error (hours since horizon step = 1h)
        "tw_peak_timing_mae_hours": float(np.mean(peak_timing_err)),
        "tw_peak_timing_median_hours": float(np.median(peak_timing_err)),
        "tw_peak_timing_p90_hours": float(np.percentile(peak_timing_err, 90)),
    }

    return {
        **overall,
        "total_water_tail": tail,
        "total_water_peaks": peaks,
        "datum": DATUM,
        "units": "metres",
        "n_windows": len(pred_water),
        "horizon_steps": pred_water.shape[1],
    }


# ---------------------------------------------------------------------------
# Per-window metrics (for event-level aggregation)
# ---------------------------------------------------------------------------

def per_window_metrics(
    pred_residual: np.ndarray,
    obs_residual: np.ndarray,
    future_tide: np.ndarray,
) -> list[dict[str, float]]:
    """Per-window dicts with: mae, rmse, corr, peak_mag_err, peak_timing_err."""
    pred_water = pred_residual + future_tide
    obs_water = obs_residual + future_tide

    results = []
    for i in range(len(pred_water)):
        pred_i = pred_water[i]
        obs_i = obs_water[i]
        err_i = pred_i - obs_i

        peak_idx_obs = int(np.argmax(obs_i))
        peak_idx_pred = int(np.argmax(pred_i))

        results.append({
            "mae": float(np.mean(np.abs(err_i))),
            "rmse": float(np.sqrt(np.mean(err_i ** 2))),
            "corr": float(np.corrcoef(pred_i, obs_i)[0, 1]) if np.std(pred_i) > 1e-8 and np.std(obs_i) > 1e-8 else float("nan"),
            "peak_mag_err": float(pred_i[peak_idx_pred] - obs_i[peak_idx_obs]),
            "peak_timing_err_hours": float(abs(peak_idx_pred - peak_idx_obs)),
        })
    return results


# ---------------------------------------------------------------------------
# Event-level aggregation + paired bootstrap CI (event-resampled)
# ---------------------------------------------------------------------------

def event_level_metrics(
    per_window: list[dict[str, float]],
    event_ids: list[str],
    station_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-window metrics to per-event and per-station means.

    Events are the correct statistical unit because windows within a
    storm are strongly correlated.
    """
    ev_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for met, eid in zip(per_window, event_ids):
        ev_groups[eid].append(met)

    ev_means: dict[str, dict[str, float]] = {}
    all_keys: set[str] = set()
    for met in per_window:
        all_keys.update(met.keys())

    for eid, wm_list in ev_groups.items():
        em = {}
        for key in all_keys:
            vals = [wm[key] for wm in wm_list if key in wm]
            clean = [v for v in vals if not np.isnan(v)]
            em[key] = float(np.mean(clean)) if clean else float("nan")
        em["n_windows"] = len(wm_list)
        ev_means[eid] = em

    result = {
        "n_events": len(ev_groups),
        "event_ids": sorted(ev_groups.keys()),
        "event_means": ev_means,
    }

    if station_ids is not None:
        st_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        for met, sid in zip(per_window, station_ids):
            st_groups[sid].append(met)

        st_means = {}
        for sid, wm_list in st_groups.items():
            sm = {}
            for key in all_keys:
                vals = [wm[key] for wm in wm_list if key in wm]
                clean = [v for v in vals if not np.isnan(v)]
                sm[key] = float(np.mean(clean)) if clean else float("nan")
            sm["n_windows"] = len(wm_list)
            st_means[sid] = sm

        result["station_means"] = st_means

    return result


def paired_event_bootstrap_ci(
    pred_water_a: np.ndarray,
    pred_water_b: np.ndarray,
    obs_water: np.ndarray,
    event_ids: list[str],
    metric_name: str = "mae",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap CI for metric(A) - metric(B) across INDEPENDENT EVENTS.

    CRITICAL: resamples EVENT IDs, NOT windows. Multiple windows from the
    same event always stay together. This preserves the independence
    assumption required for valid significance testing.

    Semantics:
        delta = metric(A) - metric(B)

        If A = candidate (e.g., A3 augmentation), B = baseline (e.g., A0 real-only):
            - delta < 0  → candidate is BETTER (lower error)
            - delta > 0  → candidate is WORSE (higher error)

    Returns dict with estimate, ci_lo, ci_hi, boot_estimate, n_boot,
    and boolean flags:
        candidate_better     = ci_hi < 0   (significantly negative delta)
        baseline_better      = ci_lo > 0   (significantly positive delta)
        neutral              = CI spans 0
    """
    rng = np.random.default_rng(seed)

    # Map each event_id to its window indices
    event_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, eid in enumerate(event_ids):
        event_to_indices[eid].append(i)

    unique_events = sorted(event_to_indices.keys())
    n_events = len(unique_events)
    assert n_events >= 5, (
        f"Too few events ({n_events}) for bootstrap — need at least 5"
    )

    def _compute_event_metric(pred, obs):
        """Compute mean metric aggregated by event."""
        per_event = []
        for eid in unique_events:
            idx = event_to_indices[eid]
            p_e = pred[idx].reshape(-1)
            o_e = obs[idx].reshape(-1)
            if metric_name == "mae":
                per_event.append(float(np.mean(np.abs(p_e - o_e))))
            elif metric_name == "rmse":
                per_event.append(float(np.sqrt(np.mean((p_e - o_e) ** 2))))
            elif metric_name == "corr":
                if np.std(p_e) > 1e-8 and np.std(o_e) > 1e-8:
                    per_event.append(float(np.corrcoef(p_e, o_e)[0, 1]))
                else:
                    per_event.append(float("nan"))
            else:
                raise ValueError(f"unknown metric_name {metric_name}")
        # Mean across events (each event is one sample)
        clean = [v for v in per_event if not np.isnan(v)]
        return float(np.mean(clean)) if clean else float("nan")

    boot_a = np.zeros(n_boot)
    boot_b = np.zeros(n_boot)
    for b in range(n_boot):
        # Resample event IDs (with replacement)
        resampled = rng.choice(unique_events, size=n_events, replace=True)
        indices = []
        for eid in resampled:
            indices.extend(event_to_indices[eid])
        idx_arr = np.array(indices)
        boot_a[b] = _compute_event_metric(pred_water_a[idx_arr], obs_water[idx_arr])
        boot_b[b] = _compute_event_metric(pred_water_b[idx_arr], obs_water[idx_arr])

    diff = boot_a - boot_b
    est = _compute_event_metric(pred_water_a, obs_water) - _compute_event_metric(pred_water_b, obs_water)
    ci_lo, ci_hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))

    return {
        "metric": metric_name,
        "delta_formula": "metric(A) - metric(B)",
        "interpretation": "delta < 0 means A has LOWER error (better for error metrics)",
        "n_events_total": n_events,
        "estimate": float(est),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "boot_estimate": float(np.mean(diff)),
        "boot_std": float(np.std(diff)),
        "n_boot": n_boot,
        "candidate_better": ci_hi < 0,   # delta is significantly negative
        "baseline_better": ci_lo > 0,    # delta is significantly positive
        "neutral": not (ci_lo > 0 or ci_hi < 0),
        "resampling_unit": "independent event_ids (NOT overlapping windows)",
    }


# ---------------------------------------------------------------------------
# Save utilities — critical for P5 reruns to enable total-water evaluation
# ---------------------------------------------------------------------------

def save_window_predictions(
    path: str | Path,
    pred_residual: np.ndarray,
    obs_residual: np.ndarray,
    future_tide: np.ndarray,
    timestamps: np.ndarray | None = None,
    station_ids: np.ndarray | None = None,
    event_ids: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save per-window predictions + metadata for reproducible total-water eval.

    This is the missing piece from historical A3 runs — they saved only
    aggregate metrics. P5 reruns MUST use this function so total-water
    evaluation is possible without re-running expensive training.

    Output file is a .npz archive:
        pred_residual:   (N, H) predicted residual targets
        obs_residual:     (N, H) observed residual targets
        future_tide:     (N, H) future FES2022 tide for each forecast
        timestamps:      (N,) origin timestamp of each forecast (optional)
        station_ids:     (N,) station ID per window (optional)
        event_ids:       (N,) canonical storm event ID per window (optional)
        meta:             JSON-serializable dict with git_sha, checkpoint paths, etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "pred_residual": pred_residual.astype(np.float32),
        "obs_residual": obs_residual.astype(np.float32),
        "future_tide": future_tide.astype(np.float32),
    }
    if timestamps is not None:
        save_dict["timestamps"] = np.array(timestamps)
    if station_ids is not None:
        save_dict["station_ids"] = np.array(station_ids)
    if event_ids is not None:
        save_dict["event_ids"] = np.array(event_ids)
    if meta is not None:
        save_dict["meta_json"] = np.array(
            json.dumps(meta, default=str), dtype=object
        )

    np.savez(path, **save_dict)


def load_window_predictions(path: str | Path) -> dict[str, Any]:
    """Inverse of save_window_predictions."""
    path = Path(path)
    assert path.exists(), f"prediction file missing: {path}"
    data = np.load(path, allow_pickle=True)
    result: dict[str, Any] = {k: data[k] for k in data.files}
    if "meta_json" in result:
        result["meta"] = json.loads(str(result.pop("meta_json")))
    return result
