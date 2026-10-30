"""Run TideTail G2 baselines on Rongjia's Prime dataset.

Reads target_timeline_v1.csv + event_table_v1.csv + train_only_metadata_v1.json
from the copied Prime dataset and evaluates persistence, seasonal and toy-GRU
on held-out test events. Real numbers only — no synthetic data in val/test.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Ensure package importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tidetail.baselines import oversampled_persistence, persistence, seasonal_baseline
from tidetail.evaluate import (
    mae,
    peak_magnitude_error,
    peak_timing_error,
    rmse,
    tail_error,
    threshold_metrics,
)

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
REGIONS = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])


def load_region(region: str):
    """Load target/events/train_metadata for a region."""
    base = DATA_ROOT / region
    # Target timeline — anchor_id, observed_total_water_level_m
    rows = []
    with open(base / "target_timeline_v1.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    # event table
    events = []
    with open(base / "event_table_v1.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            events.append(r)
    # train-only metadata
    meta = json.load(open(base / "train_only_metadata_v1.json"))
    def _fval(v):
        try: return float(v)
        except (ValueError, TypeError): return np.nan
    ts = np.array([r["timestamp_utc"] for r in rows])
    wl = np.array([_fval(r.get("water_level_m", "")) for r in rows])
    obs = np.array([_fval(r.get("observed_total_water_level_m", "")) for r in rows])
    missing = np.array([int(r.get("is_missing", "0") or "0") for r in rows])
    return {
        "region": region,
        "ts": ts, "water_level_m": wl, "observed_total": obs, "missing": missing,
        "events": events, "meta": meta,
    }


def build_windows(target: np.ndarray, horizon: int = 24, input_len: int = 72,
                  step: int = 1):
    """Sliding-window predictor-target pairs."""
    windows = []
    for start in range(0, len(target) - horizon - input_len + 1, step):
        x = target[start:start + input_len]
        y = target[start + input_len:start + input_len + horizon]
        if np.isfinite(x).all() and np.isfinite(y).all():
            windows.append((x, y))
    return windows


def evaluate_baselines(region_data, horizon=24, input_len=72):
    """Run three baselines on held-out test portion of the timeline."""
    split = region_data["meta"]
    test_start = int(split["validation_end_index"])  # after train 80% + val 10%
    target = region_data["observed_total"]  # real observed water level
    clean = np.where(region_data["missing"] == 0, target, np.nan)

    # Use non-missing test region
    train_mask = ~np.isnan(clean[:test_start])
    if train_mask.sum() < input_len + horizon:
        return {"region": region_data["region"], "status": "skip", "reason": "too-few-train"}

    test = clean[test_start:]
    test_mask = ~np.isnan(test)
    if test_mask.sum() < 100:
        return {"region": region_data["region"], "status": "skip", "reason": "too-few-test"}
    test_clean = np.where(test_mask, test, np.nan)

    # Build sliding windows on test segment (each x is input_len steps before y)
    windows = build_windows(test_clean, horizon=horizon, input_len=input_len, step=1)
    if not windows:
        return {"region": region_data["region"], "status": "skip", "reason": "no-windows"}

    X = np.stack([w[0] for w in windows])
    Y = np.stack([w[1] for w in windows])

    threshold = split["thresholds"]["total_water_q95_m"]
    results = {"region": region_data["region"], "windows": len(windows),
               "horizon": horizon, "threshold_q95_m": threshold}

    for name, pred_fn in [
        ("persistence", lambda x, h: persistence(x, h)),
        ("seasonal", lambda x, h: seasonal_baseline(x, h, period=24)),
        ("oversample", lambda x, h: oversampled_persistence(x, h, n_bootstrap=5)),
    ]:
        preds = np.stack([pred_fn(x, horizon) for x in X])
        mae_v = float(mae(Y, preds))
        rmse_v = float(rmse(Y, preds))
        tail_v = float(tail_error(Y, preds, train_threshold=threshold))
        pmag = float(np.mean([peak_magnitude_error(y, p) for y, p in zip(Y, preds)]))
        timing = float(np.mean([peak_timing_error(y, p) for y, p in zip(Y, preds)]))
        tm = threshold_metrics(Y, preds, threshold)
        results[f"{name}_mae"] = mae_v
        results[f"{name}_rmse"] = rmse_v
        results[f"{name}_tail_mae_q95"] = tail_v if np.isfinite(tail_v) else None
        results[f"{name}_peak_mag_err"] = pmag if np.isfinite(pmag) else None
        results[f"{name}_peak_timing_err"] = timing if np.isfinite(timing) else None
        results[f"{name}_threshold_recall"] = tm["recall"]
        results[f"{name}_threshold_far"] = tm["false_alarm_rate"]

    return results


if __name__ == "__main__":
    print("=== TideTail G2 Baselines on Prime 2015-2024 ===")
    print(f"Data root: {DATA_ROOT}")
    print(f"Regions: {REGIONS}")
    print()

    all_results = []
    for region in REGIONS:
        print(f"Loading {region}...")
        data = load_region(region)
        res = evaluate_baselines(data)
        all_results.append(res)

        print(f"  Train rows: {data['meta']['fit_rows']:,}  Test rows: ~{len(data['observed_total']) - int(data['meta']['validation_end_index']):,}")
        print(f"  Splits: {data['meta']['split_policy']}  q95={data['meta']['thresholds']['total_water_q95_m']:.3f}m  q99={data['meta']['thresholds']['total_water_q99_m']:.3f}m")
        if res.get("status") != "ok":
            print(f"  Skip reason: {res.get('reason')}")
        else:
            for m in ["persistence", "seasonal", "oversample"]:
                mae_v = res.get(f"{m}_mae")
                rmse_v = res.get(f"{m}_rmse")
                tail_v = res.get(f"{m}_tail_mae_q95")
                print(f"  {m:12s}  MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  tail(q95)={tail_v:.4f}" if tail_v else f"  {m:12s}  MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  tail=N/A")
        print()

    # Save results JSON
    out_path = Path(__file__).parent.parent / "data" / "baseline_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")
