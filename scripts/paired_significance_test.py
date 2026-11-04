"""Proper paired event-level significance testing for TideTail.

Corrects Wave 2.0 bootstrap which only checked candidate CI against A0 point estimate.
This script computes actual per-event paired differences and runs:
  1. Paired bootstrap of event-level differences
  2. Paired randomization test (permute group labels)
  3. Multiplicity correction
  4. Explicit confirmation/exploration labeling

Usage:
    python3 scripts/paired_significance_test.py --region south_atlantic_charleston
    python3 scripts/paired_significance_test.py --region all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
PREDS_DIR = ROOT / "data" / "predictions" / "v2"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_prediction(npz_path: Path) -> dict:
    """Load prediction file with event_ids already assigned by wave20 bootstrap."""
    with np.load(npz_path, allow_pickle=True) as d:
        tw_pred = d["pred_residual"] + d["future_tide"]
        tw_obs = d["obs_residual"] + d["future_tide"]
        errors = np.abs(tw_pred - tw_obs)
        event_ids = d["event_ids"] if "event_ids" in d else d["canonical_storm_ids"]
        meta = json.loads(d["meta_json"].item()) if "meta_json" in d else {}
    return {
        "errors": errors,
        "event_ids": event_ids,
        "meta": meta,
    }


def compute_per_unit_mae(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-event MAE. Returns (unique_event_ids, per_unit_mae, window_counts)."""
    errors = data["errors"]
    event_ids = data["event_ids"]
    unique_ids = np.sort(np.unique(event_ids))
    per_unit = np.array([
        np.mean(errors[event_ids == uid]) for uid in unique_ids
    ])
    counts = np.array([
        int(np.sum(event_ids == uid)) for uid in unique_ids
    ])
    return unique_ids, per_unit, counts


def paired_bootstrap(a_mae: np.ndarray, b_mae: np.ndarray,
                     n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap of paired differences (b - a).

    a = A0 baseline, b = A3 candidate. H0: μ(b-a) = 0.
    """
    assert len(a_mae) == len(b_mae), "Must be paired!"
    n = len(a_mae)
    diffs = b_mae - a_mae  # positive = A3 WORSE than A0 (MAE is error)

    rng = np.random.default_rng(seed)
    boot_diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[idx])

    mean_diff = np.mean(diffs)
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    # p-value: 2-sided, proportion of bootstrap differences crossing zero
    if mean_diff > 0:
        # A3 WORSE — p = proportion where boot_diff < 0
        p = 2 * np.mean(boot_diffs < 0)
    else:
        # A3 BETTER — p = proportion where boot_diff > 0
        p = 2 * np.mean(boot_diffs > 0)
    p = min(p, 1.0)

    return {
        "n_units": n,
        "mean_diff_a3_minus_a0": float(mean_diff),
        "mean_relative_diff_pct": float(mean_diff / np.mean(a_mae) * 100),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "boot_p_value_2sided": float(p),
        "diffs": diffs.tolist(),
    }


def paired_randomization_test(a_mae: np.ndarray, b_mae: np.ndarray,
                               n_perm: int = 10000, seed: int = 42) -> dict:
    """Randomization test: randomly swap labels within pairs.

    Under H0, A0 and A3 are exchangeable within each pair.
    """
    assert len(a_mae) == len(b_mae)
    n = len(a_mae)
    observed = np.mean(b_mae - a_mae)

    rng = np.random.default_rng(seed)
    perm_stats = np.zeros(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        swapped_a = np.where(swap, b_mae, a_mae)
        swapped_b = np.where(swap, a_mae, b_mae)
        perm_stats[i] = np.mean(swapped_b - swapped_a)

    # 2-sided p-value
    p = 2 * np.mean(np.abs(perm_stats) >= np.abs(observed))
    p = min(p, 1.0)

    return {
        "n_perm": n_perm,
        "observed_diff": float(observed),
        "min_perm_diff": float(np.min(perm_stats)),
        "max_perm_diff": float(np.max(perm_stats)),
        "perm_p_value_2sided": float(p),
    }


def find_baseline_predictions(region: str, model: str = "gru") -> dict[Path, dict]:
    """Find all A0 baseline prediction files for a region."""
    results = {}
    for pf in sorted(PREDS_DIR.glob(f"{region}_A0_real_only_{model}_seed*.npz")):
        seed = int(pf.stem.split("_")[-1].replace("seed", ""))
        results[seed] = pf
    return results


def find_a3_predictions(region: str) -> dict[str, Path]:
    """Find all A3 FM augmentation prediction files."""
    results = {}
    for pf in sorted(PREDS_DIR.glob(f"{region}_A3_fm_*.npz")):
        # Extract config from filename
        parts = pf.stem.split("_")
        # Something like: charleston_A3_fm_s42_ratio_0.25_gru_seed42
        results[pf.stem] = pf
    return results


def test_region(region: str, n_boot: int = 10000, n_perm: int = 10000) -> list[dict]:
    """Run all paired comparisons for a single region."""
    baselines = find_baseline_predictions(region)
    a3_files = find_a3_predictions(region)

    comparisons = []
    total_comparisons = len(a3_files) * len(baselines)

    print(f"\n{'='*80}")
    print(f"PAIRED SIGNIFICANCE: {region} — {len(baselines)} baselines × {len(a3_files)} A3 configs")
    print(f"{'='*80}")

    for a3_name, a3_path in sorted(a3_files.items()):
        # Parse A3 config
        a3_data = load_prediction(a3_path)
        a3_meta = a3_data["meta"]

        for seed, a0_path in sorted(baselines.items()):
            a0_data = load_prediction(a0_path)

            # Must use SAME event_ids for pairing
            # Wave20 bootstrap ensures same assignment
            if not np.array_equal(a0_data["event_ids"], a3_data["event_ids"]):
                print(f"  SKIP {a3_name} vs A0_seed{seed}: event_ids mismatch")
                continue

            a0_ids, a0_mae, _ = compute_per_unit_mae(a0_data)
            a3_ids, a3_mae, _ = compute_per_unit_mae(a3_data)

            # Verify exact pair matching
            if not np.array_equal(a0_ids, a3_ids):
                # Intersect
                common_ids = np.intersect1d(a0_ids, a3_ids)
                a0_mae = a0_mae[np.isin(a0_ids, common_ids)]
                a3_mae = a3_mae[np.isin(a3_ids, common_ids)]
                n_units = len(common_ids)
            else:
                n_units = len(a0_ids)

            # Run tests
            boot = paired_bootstrap(a0_mae, a3_mae, n_boot=n_boot, seed=42)
            perm = paired_randomization_test(a0_mae, a3_mae, n_perm=n_perm, seed=42)

            comparison = {
                "region": region,
                "a3_config": a3_name,
                "a0_seed": seed,
                "n_independent_units": n_units,
                "a0_mean_mae": float(np.mean(a0_mae)),
                "a3_mean_mae": float(np.mean(a3_mae)),
                "delta_pct": float((np.mean(a3_mae) - np.mean(a0_mae)) / np.mean(a0_mae) * 100),
                "bootstrap": boot,
                "randomization": perm,
            }
            comparisons.append(comparison)

            # Print immediately
            verdict_strength = ""
            if boot["boot_p_value_2sided"] < 0.05 / total_comparisons:
                verdict_strength = "BONFERRONI ✅"
            elif boot["boot_p_value_2sided"] < 0.05:
                verdict_strength = "RAW ✅"
            elif boot["mean_diff_a3_minus_a0"] > 0:
                verdict_strength = "A3 WORSE ❌"
            else:
                verdict_strength = "NOT SIGNIFICANT"

            print(f"  {a3_name:45s} vs A0_s{seed}: n={n_units:2d}, "
                  f"Δ={comparison['delta_pct']:+.1f}%, "
                  f"CI95=[{boot['bootstrap_ci_low']:.4f}, {boot['bootstrap_ci_high']:.4f}], "
                  f"p_boot={boot['boot_p_value_2sided']:.4f}, "
                  f"p_perm={perm['perm_p_value_2sided']:.4f} {verdict_strength}")

    return comparisons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="south_atlantic_charleston",
                    help="Region name or 'all'")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    regions = {
        "all": ["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
                "southeast_australia", "thames_north_sea"],
    }.get(args.region, [args.region])

    all_results = {}
    total_comparisons = 0

    for region in regions:
        results = test_region(region, n_boot=args.n_boot, n_perm=args.n_perm)
        all_results[region] = results
        total_comparisons += len(results)

    # Save
    out_path = RESULTS_DIR / "paired_significance_wave26.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved {total_comparisons} paired comparisons → {out_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — BONFERRONI α_PER_COMPARISON = 0.05 / "
          f"{total_comparisons if total_comparisons > 0 else '?'} = "
          f"{0.05/max(total_comparisons,1):.5f}")
    print(f"{'='*80}")

    bonf_alpha = 0.05 / max(total_comparisons, 1)
    sig_count = 0
    nonsig_count = 0
    worse_count = 0

    for region, results in all_results.items():
        for r in results:
            p = r["bootstrap"]["boot_p_value_2sided"]
            if p < bonf_alpha and r["bootstrap"]["mean_diff_a3_minus_a0"] < 0:
                sig_count += 1
                print(f"  ✅ {region}: {r['a3_config']} vs A0_s{r['a0_seed']} "
                      f"(p={p:.4f} < {bonf_alpha:.5f}, Δ={r['delta_pct']:+.1f}%)")
            elif r["bootstrap"]["mean_diff_a3_minus_a0"] > 0:
                worse_count += 1
            else:
                nonsig_count += 1

    print(f"\n  Survived Bonferroni: {sig_count}")
    print(f"  Not significant (A3 better but p > α): {nonsig_count}")
    print(f"  A3 WORSE than A0: {worse_count}")
    print(f"  TOTAL: {total_comparisons}")


if __name__ == "__main__":
    main()
