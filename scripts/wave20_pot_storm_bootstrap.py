"""Wave 2.0 POT+TC independent-unit bootstrap re-evaluation.

Wave 1.9 verdict (E16a): ❌ EXPLORATORY — 0-10 independent units/region.
Wave 2.0 CORRECTED framework: 6-17 independent units/region (2024 alone)
  when POT q95 + 168h Coles 2001 declustering clusters are included
  alongside canonical tropical cyclones.

This script re-runs storm-level bootstrap at the CORRECT independent-unit
level (POT cluster OR canonical TC) using:
  - data/manifests/multiyear_declustering.json  (POT cluster metadata)
  - data/predictions/v2/*.npz                    (190 prediction files, each with
                                                  timestamps + canonical_storm_ids)

Independent unit assignment priority:
  1. canonical_storm_id >= 0  → TRACK_VALIDATED_STORM
  2. POT q95 cluster (168h declustered) → OBSERVED_SURGE_EVENT
  3. observ_segment (168h gap on test windows) → BACKGROUND_PERIOD fallback

Bootstrap resampling unit: ONE independent unit (all windows within stay together).

Usage:
    python3 scripts/wave20_pot_storm_bootstrap.py

Output:
    data/results/wave20_pot_storm_bootstrap.json
    Updated event_ids in data/predictions/v2/*.npz (backed up)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
PREDS_DIR = ROOT / "data" / "predictions" / "v2"
DECLUSTERING = ROOT / "data" / "manifests" / "multiyear_declustering.json"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_ts_to_ns(ts_str: str) -> int:
    """Parse ISO timestamp to nanoseconds since epoch."""
    t = pd.Timestamp(ts_str)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return int(t.value)  # nanoseconds


def load_pot_clusters(region: str, year: int, declustering: dict) -> list[dict]:
    """Load POT clusters for region/year from manifest."""
    clusters = declustering.get(region, {}).get("pot_clusters", {}).get(str(year), [])
    result = []
    for c in clusters:
        result.append({
            "cluster_id": c["cluster_id"],
            "start_ns": parse_ts_to_ns(c["start"]),
            "end_ns": parse_ts_to_ns(c["end"]),
            "peak_residual_m": c["peak_residual_m"],
            "exceed_count": c["exceed_count"],
            "threshold_q95": c["threshold_q95"],
        })
    return result


def assign_pot_units(timestamps_ns: np.ndarray, pot_clusters_2024: list[dict],
                     canonical_ids: np.ndarray) -> np.ndarray:
    """Assign each test window to an independent unit.

    Priority:
      1. canonical_storm_id >= 0 → use canonical_id (positive)
      2. POT cluster match → negative -(cluster_id + 1) (unique per region)
      3. fallback → -999 (background, non-extreme)

    Returns combined_units array (same length as timestamps_ns).
    """
    N = len(timestamps_ns)
    combined = np.full(N, -999, dtype=np.int64)

    # Canonical storms first (highest priority)
    tc_mask = canonical_ids >= 0
    combined[tc_mask] = canonical_ids[tc_mask]

    # Then POT clusters for non-TC windows
    non_tc = ~tc_mask
    for c in pot_clusters_2024:
        c_start = c["start_ns"]
        c_end = c["end_ns"]
        # Match windows within cluster's start-end range
        pot_mask = (timestamps_ns >= c_start) & (timestamps_ns <= c_end) & non_tc
        combined[pot_mask] = -(c["cluster_id"] + 1)

    return combined


def storm_level_bootstrap(tw_pred: np.ndarray, tw_obs: np.ndarray,
                          group_ids: np.ndarray, n_boot: int = 2000,
                          seed: int = 42) -> dict:
    """Bootstrap at INDEPENDENT UNIT level.

    Each independent unit is resampled as a whole — all windows within stay
    together. This preserves within-unit correlation while correctly treating
    cross-unit variation as the uncertainty source.
    """
    rng = np.random.default_rng(seed)
    errors = np.abs(tw_pred - tw_obs)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    # Per-unit MAE (mean absolute error within the unit)
    group_means = np.array([np.mean(errors[group_ids == g]) for g in unique_groups])

    # Per-unit window count (for power estimation weights)
    group_sizes = np.array([int(np.sum(group_ids == g)) for g in unique_groups])

    # Bootstrap: resample units with replacement
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        boot[b] = np.mean(group_means[chosen])

    return {
        "independent_units": int(n_groups),
        "total_test_windows": len(group_ids),
        "point_mae": float(np.mean(group_means)),
        "bootstrap_mean": float(np.mean(boot)),
        "ci_low_2.5": float(np.percentile(boot, 2.5)),
        "ci_high_97.5": float(np.percentile(boot, 97.5)),
        "ci_low_5": float(np.percentile(boot, 5)),
        "ci_high_95": float(np.percentile(boot, 95)),
        "ci_width": float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)),
        "unit_mae_std": float(np.std(group_means)) if n_groups > 1 else 0.0,
        "unit_window_counts": group_sizes.tolist(),
        "unique_unit_ids": unique_groups.tolist(),
        "n_boot": n_boot,
    }


def power_diagnostic(units_per_region: dict) -> str:
    """Generate power assessment string per region."""
    lines = []
    for region, n_units in units_per_region.items():
        if n_units >= 40:
            verdict = "POWERED (can detect ~5% Δ MAE)"
        elif n_units >= 25:
            verdict = "POWERED (can detect ~10% Δ MAE)"
        elif n_units >= 12:
            verdict = "POWERED (can detect ~15% Δ MAE)"
        elif n_units >= 8:
            verdict = "BORDERLINE (exploratory only for <15% Δ)"
        else:
            verdict = "UNDER-POWERED (descriptive only)"
        lines.append(f"  {region}: {n_units} independent units → {verdict}")
    return "\n".join(lines)


def process_region(region: str, declustering: dict):
    print(f"\n{'='*70}")
    print(f"REGION: {region}")
    print(f"{'='*70}")

    # Load POT clusters for test year 2024
    pot_2024 = load_pot_clusters(region, 2024, declustering)
    print(f"  POT clusters 2024 (q95 + 168h declustered): {len(pot_2024)}")
    for c in pot_2024:
        print(f"    cluster {c['cluster_id']}: {pd.Timestamp(c['start_ns']).date()} → "
              f"{pd.Timestamp(c['end_ns']).date()}, peak={c['peak_residual_m']:.3f}m, "
              f"exceeds={c['exceed_count']}")

    # Process each prediction file
    region_files = sorted(PREDS_DIR.glob(f"{region}_*_seed*.npz"))
    print(f"\n  Prediction files: {len(region_files)}")

    # Build backup dir
    backup_dir = PREDS_DIR / "v2_before_wave20_pot"
    backup_dir.mkdir(exist_ok=True)

    results = {}
    unit_counts = {}  # track independent units per group

    for pf in region_files:
        # Backup first time only
        if not (backup_dir / pf.name).exists():
            shutil.copy2(pf, backup_dir / pf.name)

        with np.load(pf, allow_pickle=True) as d:
            tw_pred = d["pred_residual"] + d["future_tide"]
            tw_obs = d["obs_residual"] + d["future_tide"]
            timestamps_ns = d["timestamps"]
            canonical_ids = d["canonical_storm_ids"]
            meta = json.loads(d["meta_json"].item()) if "meta_json" in d else {}
            data_dict = dict(d)

        # Assign combined independent units using POT+TC framework
        combined_units = assign_pot_units(timestamps_ns, pot_2024, canonical_ids)

        # Bootstrap on combined independent units
        bs = storm_level_bootstrap(tw_pred, tw_obs, combined_units, n_boot=2000, seed=42)

        # Update npz with new event_ids
        data_dict["event_ids"] = combined_units
        data_dict["meta_json"] = np.array([json.dumps({
            **meta,
            "independent_units_wave20": bs["independent_units"],
            "pot_clusters_2024": len(pot_2024),
            "bootstrap_wave20": bs,
            "backfill_version": "wave20_pot_tc_bootstrap_v1",
            "declustering_method": "Coles 2001 POT q95 + 168h gap + canonical TC overlay",
        })], dtype=object)
        np.savez(pf, **data_dict)

        # Parse group/model/seed from meta or filename
        cfg = meta.get("group") or pf.stem.split("_")[1]
        model = meta.get("model") or pf.stem.split("_")[2] if len(pf.stem.split("_")) > 2 else "?"
        seed = meta.get("seed") or 42
        key = f"{cfg} {model} seed{seed}"
        results[key] = bs
        unit_counts[cfg] = bs["independent_units"]

    # Print summary
    a0_key = None
    for k in results:
        if "A0_real_only gru seed42" in k:
            a0_key = k
            break

    if a0_key:
        a0 = results[a0_key]
        print(f"\n  A0 GRU seed42: MAE={a0['point_mae']:.4f}, "
              f"CI95=[{a0['ci_low_5']:.4f},{a0['ci_high_95']:.4f}], "
              f"units={a0['independent_units']}")
        print(f"  Unit-level MAE std: {a0['unit_mae_std']:.4f}")

        verdicts = {"better": 0, "worse": 0, "overlap": 0}
        for cfg, bs in results.items():
            if ("A3" in cfg or "A1" in cfg or "A2" in cfg) and "gru" in cfg:
                delta = (bs["point_mae"] - a0["point_mae"]) / a0["point_mae"] * 100
                # Significance: candidate CI entirely below/above A0's point estimate
                if bs["ci_high_97.5"] < a0["point_mae"] - 1e-6:
                    verdicts["better"] += 1
                    v = "✅ SIGNIFICANTLY BETTER"
                elif bs["ci_low_2.5"] > a0["point_mae"] + 1e-6:
                    verdicts["worse"] += 1
                    v = "❌ SIGNIFICANTLY WORSE"
                else:
                    verdicts["overlap"] += 1
                    v = "➖ NOT SIGNIFICANT"
                print(f"  {v} {cfg}: Δ%={delta:+.2f}, MAE={bs['point_mae']:.4f}, "
                      f"CI95=[{bs['ci_low_5']:.4f},{bs['ci_high_95']:.4f}], "
                      f"units={bs['independent_units']}")

        print(f"  Totals: better={verdicts['better']}, worse={verdicts['worse']}, overlap={verdicts['overlap']}")

    return {
        "region": region,
        "pot_clusters_2024": len(pot_2024),
        "total_independent_units_A0": results.get("A0_real_only gru seed42", {}).get("independent_units", 0),
        "bootstrap_results": results,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Wave 2.0 POT+TC Independent-Unit Bootstrap Re-evaluation")
    print("Independent unit = POT q95 cluster (168h gap) OR canonical tropical cyclone")
    print("=" * 70)

    declustering = json.loads(DECLUSTERING.read_text())
    print(f"\nLoaded declustering manifest: {DECLUSTERING}")
    print(f"Regions: {list(declustering.keys())}")

    # First show power diagnostic for 2024
    print("\n" + "=" * 70)
    print("POWER DIAGNOSTIC (2024 test year, POT+TC)")
    print("=" * 70)
    for region in declustering:
        pot_n = len(declustering[region]["pot_clusters"].get("2024", []))
        # Canonical TCs 2024
        tcs_2024 = declustering[region].get("tcs", {}).get("2024", [])
        tc_n = len(tcs_2024) if isinstance(tcs_2024, list) else 0
        total = pot_n + tc_n
        if total >= 12:
            verdict = "POWERED (≥12 units)"
        elif total >= 8:
            verdict = "BORDERLINE"
        else:
            verdict = "UNDER-POWERED"
        print(f"  {region}: POT={pot_n} + TC={tc_n} = {total} → {verdict}")

    all_results = []
    for r in ["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
              "southeast_australia", "thames_north_sea"]:
        all_results.append(process_region(r, declustering))

    # Save
    out = RESULTS_DIR / "wave20_pot_storm_bootstrap.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))

    # Print overall verdicts
    print(f"\n{'='*80}")
    print("OVERALL WAVE 2.0 VERDICTS")
    print(f"{'='*80}")
    print("E16a A3 FM augmentation region-dependent, assessed at POT+TC independent-unit level:")
    print()

    for rdata in all_results:
        region = rdata["region"]
        units = rdata["total_independent_units_A0"]
        a0_bs = rdata["bootstrap_results"].get("A0_real_only gru seed42")
        if not a0_bs:
            print(f"  {region}: NO A0 baseline found")
            continue
        a0_mae = a0_bs["point_mae"]
        a0_ci_low = a0_bs["ci_low_5"]
        a0_ci_high = a0_bs["ci_high_95"]

        # Best A3 comparison (group with lowest point MAE)
        best_a3 = None
        best_delta = 999
        for cfg, bs in rdata["bootstrap_results"].items():
            if "A3" in cfg and "gru" in cfg:
                delta = (bs["point_mae"] - a0_mae) / a0_mae * 100
                if delta < best_delta:
                    best_delta = delta
                    best_a3 = (cfg, bs)

        if best_a3:
            cfg, bs = best_a3
            delta = best_delta
            # Determine verdict
            if bs["ci_high_97.5"] < a0_mae - 1e-6:
                verdict = "✅ SIGNIFICANT IMPROVEMENT"
            elif bs["ci_low_2.5"] > a0_mae + 1e-6:
                verdict = "❌ SIGNIFICANT HARM"
            elif delta < -3:
                verdict = "📉 NUMERIC IMPROVEMENT (CI overlaps)"
            elif delta > 3:
                verdict = "📈 NUMERIC HARM (CI overlaps)"
            else:
                verdict = "➖ NEGLIGIBLE"

            print(f"  {region:28s}: units={units:2d}, best A3={verdict:30s} Δ%={delta:+.1f}")

    print(f"\nResults saved → {out}")
    print("Backups →", PREDS_DIR / "v2_before_wave20_pot")
