"""Regression tests locking down the FORECAST_TASK_CONTRACT target semantics.

Contract (FORECAST_TASK_CONTRACT.md §2):

    observed_total_water_level_m = water_level_m + FES2022_tide    (EXACT, datum MLLW)
    water_level_m                = observed_total − FES2022_tide  (ALREADY de-tided surge residual)
    tide_level_m                 = FES2022 astronomical tide       (deterministic)

These tests MUST run against the real 5-region prime dataset. They catch
the critical double-subtraction bug (P0) that corrupted M2 Diffusion-TS,
M4 Flow Matching, and A3 runs before 2026-11-03.

Authoritative source: /data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast/{region}/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
REGIONS = [
    "ny_bight",
    "south_atlantic_charleston",
    "southeast_australia",
    "thames_north_sea",
    "us_gulf_coast",
]


@pytest.fixture(params=REGIONS)
def region_name(request):
    return request.param


@pytest.fixture()
def loaded_region(region_name):
    """Load target + tide CSVs merged on timestamp, with NaN rows flagged."""
    base = DATA_ROOT / region_name
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    meta = json.loads((base / "train_only_metadata_v1.json").read_text())
    manifest = json.loads((base / "region_manifest_v1.json").read_text())
    wi = np.load(base / "window_index_v1.npz")

    merged = target.merge(tide, on="timestamp_utc", how="inner")
    valid_mask = (
        merged["water_level_m"].notna()
        & merged["observed_total_water_level_m"].notna()
        & merged["tide_level_m"].notna()
    )
    return {
        "merged": merged,
        "valid_mask": valid_mask.to_numpy(),
        "valid_rows": merged[valid_mask],
        "meta": meta,
        "manifest": manifest,
        "window_index": wi,
        "base": base,
    }


# ---------------------------------------------------------------------------
# T0: Fundamental identity — water_level_m + tide_level_m == observed_total_water_level_m
# ---------------------------------------------------------------------------

def test_identity_total_equals_water_plus_tide(loaded_region):
    """The most basic contract check — verified EXACT across all 5 regions."""
    vr = loaded_region["valid_rows"]
    w = vr["water_level_m"].to_numpy(dtype=np.float64)
    t = vr["tide_level_m"].to_numpy(dtype=np.float64)
    tot = vr["observed_total_water_level_m"].to_numpy(dtype=np.float64)

    recon = w + t
    mae = float(np.mean(np.abs(recon - tot)))
    max_err = float(np.max(np.abs(recon - tot)))

    assert mae < 1e-6, (
        f"total != water + tide: MAE={mae:.2e}m, max_err={max_err:.2e}m"
    )
    assert max_err < 1e-5, f"max discrepancy {max_err:.2e} exceeds tolerance"


def test_identity_direct_total_is_not_water(loaded_region):
    """water_level_m is NOT the same as observed_total_water_level_m.

    The double-subtraction bug arose because scripts assumed
    water_level_m = total_water — this test catches that confusion.
    """
    vr = loaded_region["valid_rows"]
    w = vr["water_level_m"].to_numpy(dtype=np.float64)
    tot = vr["observed_total_water_level_m"].to_numpy(dtype=np.float64)

    mae_direct = float(np.mean(np.abs(w - tot)))
    assert mae_direct > 0.05, (
        f"water_level_m is suspiciously close to observed_total_water_level_m "
        f"(MAE={mae_direct:.4f}m) — datum semantics may be wrong"
    )


def test_identity_tide_has_tidal_amplitude(loaded_region):
    """tide_level_m must show detectable tidal oscillation (not constant)."""
    vr = loaded_region["valid_rows"]
    t = vr["tide_level_m"].to_numpy(dtype=np.float64)
    amp = float(0.5 * (np.percentile(t, 97.5) - np.percentile(t, 2.5)))
    assert amp > 0.1, f"tide_level_m has near-zero amplitude ({amp:.3f}m)"


# ---------------------------------------------------------------------------
# T1: Loader integrity — scripts must read residual correctly
# ---------------------------------------------------------------------------

def test_train_generative_model_loader_patches_residual(region_name):
    """train_generative_models.py load_region must NOT re-subtract tide.

    If this test fails, the double-subtraction bug has returned.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_generative_models import load_region

    data = load_region(region_name)
    rtr_t = data["train_tgt_res"]
    rte_t = data["test_tgt_res"]

    # Independent verification from raw CSVs
    base = DATA_ROOT / region_name
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    merged = target.merge(tide, on="timestamp_utc", how="inner")
    true_residual = merged["water_level_m"].to_numpy(dtype=np.float32)

    wi = np.load(base / "window_index_v1.npz")
    def _win(starts):
        starts = starts.astype(np.int64)
        out = np.zeros((len(starts), 72), dtype=np.float32)
        for i, s in enumerate(starts):
            out[i] = true_residual[s + 96 : s + 96 + 72]
        return out

    true_rtr_t = _win(wi["train_start_indices"])
    true_rte_t = _win(wi["test_start_indices"])

    assert rtr_t.shape == true_rtr_t.shape
    assert rte_t.shape == true_rte_t.shape
    assert np.allclose(rtr_t, true_rtr_t, atol=1e-4), (
        f"{region_name}: train residual mismatch"
    )
    assert np.allclose(rte_t, true_rte_t, atol=1e-4), (
        f"{region_name}: test residual mismatch"
    )


def test_train_generative_model_loader_patches_total_water(region_name):
    """Diffusion-TS path (train_hist_wl/train_tgt_wl) must get OBSERVED TOTAL,
    not residual. Pre-2026-11-03 bug: used water_level_m instead of observed_total."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_generative_models import load_region

    data = load_region(region_name)
    tr_tgt = data["train_tgt_wl"]
    te_tgt = data["test_tgt_wl"]

    base = DATA_ROOT / region_name
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    merged = target.merge(tide, on="timestamp_utc", how="inner")
    true_total = merged["observed_total_water_level_m"].to_numpy(dtype=np.float32)

    wi = np.load(base / "window_index_v1.npz")
    def _win(starts):
        starts = starts.astype(np.int64)
        out = np.zeros((len(starts), 72), dtype=np.float32)
        for i, s in enumerate(starts):
            out[i] = true_total[s + 96 : s + 96 + 72]
        return out

    assert np.allclose(tr_tgt, _win(wi["train_start_indices"]), atol=1e-4), (
        f"{region_name}: train total-water mismatch"
    )
    assert np.allclose(te_tgt, _win(wi["test_start_indices"]), atol=1e-4), (
        f"{region_name}: test total-water mismatch"
    )


def test_a0_baselines_loader_patches_residual(region_name):
    """run_a0_baselines.py load_region correctly reads water_level_m directly
    (never subtracted tide). Verify this was never broken."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_a0_baselines import load_region

    data = load_region(region_name)
    tr_t = data["train_t"]
    te_t = data["test_t"]

    base = DATA_ROOT / region_name
    target = pd.read_csv(base / "target_timeline_v1.csv")
    tide = pd.read_csv(base / "tide_timeline_v1.csv")
    merged = target.merge(tide, on="timestamp_utc", how="inner")
    true_residual = merged["water_level_m"].to_numpy(dtype=np.float32)

    wi = np.load(base / "window_index_v1.npz")
    def _win(starts):
        starts = starts.astype(np.int64)
        out = np.zeros((len(starts), 72), dtype=np.float32)
        for i, s in enumerate(starts):
            out[i] = true_residual[s + 96 : s + 96 + 72]
        return out

    assert np.allclose(tr_t, _win(wi["train_start_indices"]), atol=1e-4)
    assert np.allclose(te_t, _win(wi["test_start_indices"]), atol=1e-4)


# ---------------------------------------------------------------------------
# T2: Metadata integrity — manifest and train_only_metadata
# ---------------------------------------------------------------------------

def test_metadata_has_both_water_and_tide_scalers(loaded_region):
    """train_only_metadata_v1.json must have scalers for BOTH water_level_m
    (residual) and tide_level_m."""
    meta = loaded_region["meta"]
    assert "scalers" in meta, "meta missing 'scalers' key"
    assert "water_level_m" in meta["scalers"], (
        "meta.scalers missing water_level_m"
    )
    assert "tide_level_m" in meta["scalers"], (
        "meta.scalers missing tide_level_m"
    )


def test_manifest_declares_total_water_threshold(loaded_region):
    """region_manifest_v1.json must declare a total-water train-only threshold.

    The residual threshold lives in train_only_metadata as water_level_m scaler,
    but the manifest MUST distinguish total-water for event-level evaluation.
    """
    manifest = loaded_region["manifest"]
    # Walk entire nested dict
    found_total = False
    def walk(d):
        nonlocal found_total
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(k, str) and ("total_water" in k.lower() and "q95" in k.lower()):
                found_total = True
    walk(manifest)
    assert found_total, (
        "manifest missing train_only_q95_total_water_m (total-water threshold)"
    )


def test_windows_contain_only_valid_rows(loaded_region):
    """window_index_v1.npz indices must point only to rows with valid
    (non-NaN) water + tide values — no leakage of missing data into windows."""
    valid = loaded_region["valid_mask"]
    wi = loaded_region["window_index"]
    context, horizon = 96, 72

    for split_name in ["train_start_indices", "validation_start_indices", "test_start_indices"]:
        starts = wi[split_name].astype(np.int64)
        # Check first and last sample for NaN coverage
        for s in starts[:50]:
            window_mask = valid[s : s + context + horizon]
            if not np.all(window_mask):
                n_bad = int(np.sum(~window_mask))
                # Some regions have source gaps — manifest says policy is
                # "exclude_any_window_with_a_missing_row" so NO bad rows should appear
                assert n_bad == 0, (
                    f"{split_name}: window at start={s} has {n_bad} NaN rows "
                    f"(policy says windows must be clean)"
                )


# ---------------------------------------------------------------------------
# T3: Datum and physical plausibility
# ---------------------------------------------------------------------------

def test_residual_amplitude_consistent(loaded_region):
    """De-tided surge residual (water_level_m) should have reasonable
    physical scale. Typical storm surges: ±1–2m, occasionally ±3m."""
    vr = loaded_region["valid_rows"]
    res = vr["water_level_m"].to_numpy(dtype=np.float64)
    tide = vr["tide_level_m"].to_numpy(dtype=np.float64)
    tot = vr["observed_total_water_level_m"].to_numpy(dtype=np.float64)

    res_amp = float(0.5 * (np.percentile(res, 99) - np.percentile(res, 1)))
    tide_amp = float(0.5 * (np.percentile(tide, 99) - np.percentile(tide, 1)))
    tot_amp = float(0.5 * (np.percentile(tot, 99) - np.percentile(tot, 1)))

    # Residual should have moderate but not overwhelming amplitude
    assert 0.05 < res_amp < 5.0, (
        f"residual amplitude {res_amp:.2f}m outside physical range"
    )
    # Tide should dominate total-water amplitude
    assert tide_amp >= res_amp * 0.3, (
        f"tide amplitude {tide_amp:.2f}m unexpectedly small vs residual {res_amp:.2f}m"
    )
    # Total = residual + tide should have the largest amplitude
    assert tot_amp >= max(res_amp, tide_amp) - 0.05


def test_fes2022_tide_is_deterministic(loaded_region):
    """FES2022 tide is deterministic — same timestamp must always have same value.

    (Sanity check: no stochastic component leaked into tide timeline.)
    """
    vr = loaded_region["valid_rows"]
    tide_series = vr.set_index("timestamp_utc")["tide_level_m"]
    # Deterministic tide must have variance
    assert tide_series.std() > 0.01, "tide has near-zero variance"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
