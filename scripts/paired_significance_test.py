"""Wave 2.7 paired event-level significance testing (corrected).

Fixes two defects in the Wave 2.6 script:

1. It paired every A3 candidate against a **GRU** A0 baseline, including TCN
   candidates. That confounds architecture with augmentation. Here candidates
   are paired only against A0 runs of the SAME forecaster.
2. It silently included the background unit (``-999``, ~6k of 8.6k windows) in
   the event-level mean. Here the primary cohort is **extreme units only**
   (``event_id != -999``); the all-units result is retained as secondary.

Multiplicity is reported for the explicit family actually tested, and the
claim is labelled exploratory unless it survives the family-wide correction.

Usage:
    python3 scripts/paired_significance_test.py --region all

Outputs:
    data/results/paired_significance_wave27.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PREDS_DIR = ROOT / "data" / "predictions" / "v2"
RESULTS_DIR = ROOT / "data" / "results"

REGIONS = [
    "ny_bight",
    "us_gulf_coast",
    "south_atlantic_charleston",
    "southeast_australia",
    "thames_north_sea",
]
AUGMENT_GROUPS = ("A1_oversampling", "A2_history_noise")


def load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        # target semantics: total = residual + tide
        errors = np.abs(
            (d["pred_residual"] + d["future_tide"]) - (d["obs_residual"] + d["future_tide"])
        )
        event_ids = d["event_ids"].astype(np.int64)
        meta = json.loads(d["meta_json"].item()) if "meta_json" in d else {}
    return {"errors": errors, "event_ids": event_ids, "meta": meta}


def per_unit_mae(errors: np.ndarray, ids: np.ndarray, extreme_only: bool) -> tuple[np.ndarray, np.ndarray]:
    keep = ids != -999 if extreme_only else np.ones(len(ids), dtype=bool)
    e, g = errors[keep], ids[keep]
    unique = np.sort(np.unique(g))
    mae = np.array([e[g == u].mean() for u in unique])
    counts = np.array([int((g == u).sum()) for u in unique])
    return unique, mae, counts


def paired_bootstrap(diffs: np.ndarray, n_boot: int, seed: int) -> dict:
    """Vectorised paired bootstrap.

    The Monte-Carlo resolution must be fine enough to reach the family
    Bonferroni threshold: with ``n_boot`` resamples the smallest attainable
    two-sided p-value is ``2/(n_boot+1)``. For a 320-comparison family
    (alpha = 1.56e-4) we therefore need ``n_boot >= ~13,000``.
    """
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    observed = float(diffs.mean())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    # Tail mass in the direction opposite to the observed effect.
    tail = float(np.mean(boot < 0)) if observed > 0 else float(np.mean(boot > 0))
    hits = tail * n_boot          # count-style, NOT a proportion
    # bounded Monte-Carlo two-sided p-value; never report a literal zero
    p = min(1.0, 2.0 * (hits + 1.0) / (n_boot + 1.0))
    return {
        "mean_diff_a3_minus_a0": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "p_value": float(p),
        "n_boot": n_boot,
        "resolution_floor": float(2.0 / (n_boot + 1.0)),
    }


def paired_randomization(diffs: np.ndarray, n_perm: int, seed: int) -> dict:
    """Vectorised paired sign-flip randomization test."""
    n = len(diffs)
    observed = abs(float(diffs.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    hits = int(np.sum(np.abs((diffs * signs).mean(axis=1)) >= observed))
    p = (hits + 1.0) / (n_perm + 1.0)
    return {"observed_abs_diff": observed, "p_value": p, "n_perm": n_perm,
            "resolution_floor": float(1.0 / (n_perm + 1.0))}


def discover(region: str) -> tuple[dict, dict]:
    baselines: dict[str, dict[int, Path]] = {}
    candidates: dict[str, Path] = {}
    for path in sorted(PREDS_DIR.glob(f"{region}_*.npz")):
        if path.stem.endswith(("persistence", "seasonal_24h")):
            continue
        meta = load_prediction(path)["meta"]
        group = meta.get("group", "")
        model = meta.get("model")
        seed = meta.get("seed")
        if group == "A0_real_only" and model and seed is not None:
            baselines.setdefault(model, {})[int(seed)] = path
        elif group in AUGMENT_GROUPS or group.startswith("A3_fm") or group.startswith("A3_flow_matching"):
            candidates[path.stem] = path
    return baselines, candidates


def compare_region(region: str, n_boot: int, n_perm: int) -> dict:
    baselines, candidates = discover(region)
    if not baselines:
        return {"region": region, "comparisons": [], "note": "no A0 baselines"}

    comparisons = []
    for name, path in candidates.items():
        cand = load_prediction(path)
        model = cand["meta"].get("model")
        if model not in baselines:
            continue
        for a0_seed, a0_path in sorted(baselines[model].items()):
            a0 = load_prediction(a0_path)
            if len(a0["event_ids"]) != len(cand["event_ids"]):
                continue
            if not np.array_equal(a0["event_ids"], cand["event_ids"]):
                continue

            record = {
                "region": region,
                "candidate": name,
                "candidate_group": cand["meta"].get("group"),
                "model": model,
                "candidate_train_seed": cand["meta"].get("seed"),
                "baseline_seed": a0_seed,
            }
            for cohort, extreme_only in (("extreme_units", True), ("all_units", False)):
                a0_ids, a0_mae, _ = per_unit_mae(a0["errors"], a0["event_ids"], extreme_only)
                c_ids, c_mae, _ = per_unit_mae(cand["errors"], cand["event_ids"], extreme_only)
                assert np.array_equal(a0_ids, c_ids)
                diffs = c_mae - a0_mae  # >0 means augmentation is worse
                record[cohort] = {
                    "n_units": int(len(a0_ids)),
                    "a0_mean_mae": float(a0_mae.mean()),
                    "a3_mean_mae": float(c_mae.mean()),
                    "delta_pct": float(diffs.mean() / a0_mae.mean() * 100.0),
                    "bootstrap": paired_bootstrap(diffs, n_boot, 42),
                    "randomization": paired_randomization(diffs, n_perm, 42),
                }
            comparisons.append(record)

    return {
        "region": region,
        "baselines_per_model": {m: sorted(s) for m, s in baselines.items()},
        "n_candidates": len(candidates),
        "comparisons": comparisons,
    }


def annotate(result: dict, family_size: int, alpha_family: float) -> dict:
    survivors = []
    for record in result["comparisons"]:
        ext = record["extreme_units"]
        p_boot = ext["bootstrap"]["p_value"]
        p_perm = ext["randomization"]["p_value"]
        improves = ext["bootstrap"]["mean_diff_a3_minus_a0"] < 0
        record["directional_improvement"] = bool(improves)
        record["raw_p_boot"] = p_boot
        record["raw_p_perm"] = p_perm
        record["ci_excludes_zero"] = bool(ext["bootstrap"].get("ci_excludes_zero", False))
        record["bonferroni_alpha"] = alpha_family
        record["bonferroni_significant_boot"] = bool(p_boot < alpha_family and improves)
        record["bonferroni_significant_perm"] = bool(p_perm < alpha_family and improves)
        # Survivor requires BOTH tests, a CI excluding zero, and an improvement.
        if (record["bonferroni_significant_boot"] and record["bonferroni_significant_perm"]
                and record["ci_excludes_zero"]):
            survivors.append(record["candidate"] + f"|A0seed{record['baseline_seed']}")
    result["family_size"] = family_size
    result["family_bonferroni_alpha"] = alpha_family
    result["bonferroni_survivors"] = survivors
    result["verdict"] = (
        "EXPLORATORY — survives within-region family correction"
        if survivors
        else "NO CONFIRMATORY EVIDENCE under family-wide correction"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="all")
    ap.add_argument("--n-boot", type=int, default=30000,
                    help="bootstrap resamples; floor 2/(n+1) must be below alpha")
    ap.add_argument("--n-perm", type=int, default=30000,
                    help="sign-flip permutations; floor 1/(n+1) must be below alpha")
    args = ap.parse_args()

    regions = REGIONS if args.region == "all" else [args.region]
    results = {}
    total = 0
    for region in regions:
        res = compare_region(region, args.n_boot, args.n_perm)
        results[region] = res
        total += len(res.get("comparisons", []))

    # Family = every candidate-vs-baseline comparison actually tested.
    alpha = 0.05 / total if total else float("nan")
    print("=" * 78)
    print(f"PAIRED SIGNIFICANCE (within-architecture, extreme-unit primary)")
    print(f"Total comparisons tested (family): {total}   Bonferroni alpha: {alpha:.6f}")
    print("=" * 78)

    all_survivors = []
    for region, res in results.items():
        res = annotate(res, total, alpha)
        print(f"\n{region}: {len(res.get('comparisons', []))} comparisons")
        for rec in res["comparisons"]:
            ext = rec["extreme_units"]
            flag = "SIG" if rec["bonferroni_significant_boot"] and rec["bonferroni_significant_perm"] else "   "
            print(
                f"  [{flag}] {rec['model']:3s} {rec['candidate_group']:24s} "
                f"{rec['candidate']:48s} A0s{rec['baseline_seed']} "
                f"n={ext['n_units']:2d} d={ext['delta_pct']:+6.1f}% "
                f"p_boot={rec['raw_p_boot']:.5f} p_perm={rec['raw_p_perm']:.5f}"
            )
            if flag == "SIG":
                all_survivors.append((region, rec["candidate"], rec["baseline_seed"]))
        print(f"  survivors: {res['bonferroni_survivors']}")

    # Baseline-seed robustness: a finding that survives against only one A0
    # seed is confounded by baseline quality, not augmentation.
    from collections import defaultdict
    by_candidate = defaultdict(list)
    for region, res in results.items():
        for rec in res.get("comparisons", []):
            by_candidate[(region, rec["candidate"])].append(rec)
    robust = []
    seed_fragile = []
    for key, recs in by_candidate.items():
        sig = [r for r in recs if r["bonferroni_significant_boot"]
               and r["bonferroni_significant_perm"] and r["ci_excludes_zero"]
               and r["directional_improvement"]]
        if recs and len(sig) == len(recs):
            robust.append(key)
        elif sig:
            seed_fragile.append([key[0], key[1],
                                 [r["baseline_seed"] for r in sig],
                                 [r["baseline_seed"] for r in recs]])
    print(f"\nRobust across all A0 seeds: {len(robust)}   seed-fragile: {len(seed_fragile)}")
    seed_robust_survivors = [[r, c] for r, c in robust]
    for item in seed_fragile:
        print(f"  seed-fragile: {item[0]} {item[1]} sig_vs={item[2]} of {item[3]}")
    print("\nArchitecture summary (extreme units, directional):")
    arch = {}
    for model in ("gru", "tcn"):
        n = better = worse = 0
        deltas = []
        for res in results.values():
            for rec in res.get("comparisons", []):
                if rec["model"] != model:
                    continue
                n += 1
                deltas.append(rec["extreme_units"]["delta_pct"])
                if rec["extreme_units"]["delta_pct"] < 0:
                    better += 1
                else:
                    worse += 1
        if n:
            print(f"  {model}: n={n}, improved={better}, worsened={worse}, "
                  f"mean_delta={np.mean(deltas):+.2f}%")
            arch[model] = {"n": n, "improved": better, "worsened": worse,
                           "mean_delta_pct": float(np.mean(deltas),
                                                   )}
            print(f"  {model} by region: " + ", ".join(
                f"{r}={np.mean([x['extreme_units']['delta_pct'] for x in res['comparisons'] if x['model']==model]):+.2f}%"
                for r, res in results.items()))

    payload = {
        "audit_version": "wave27_v1",
        "primary_cohort": "extreme independent units (event_id != -999)",
        "pairing": "within-architecture (candidate model == A0 model)",
        "family_size": total,
        "bonferroni_alpha": alpha,
        "regions": results,
        "architecture_summary": arch,
        "survivors_global": seed_robust_survivors,
        "seed_fragile_survivors": seed_fragile,
        "nominal_survivors_ignoring_baseline_seed": all_survivors,
        "resolution_floor_boot": 2.0 / (args.n_boot + 1.0),
        "resolution_floor_perm": 1.0 / (args.n_perm + 1.0),
        "resolution_sufficient": bool(
            2.0 / (args.n_boot + 1.0) <= alpha and 1.0 / (args.n_perm + 1.0) <= alpha),
        "coverage_note": (
            "Family covers all A1/A2/A3_fm candidate runs available in "
            "data/predictions/v2 for the listed regions: A1 oversampling (2 seeds), "
            "A2 history noise (2 seeds), A3 FM (2 generator seeds x ratios 0.1/0.25/0.5), "
            "each paired against A0 of the SAME forecaster (2 seeds). That is "
            "16 candidates x 2 baselines x 2 architectures x 5 regions = 320 tests. "
            "It does NOT enumerate the full historical search space: tail-aware A3 "
            "predictions are absent, and discarded ratios, generator hyperparameters "
            "and model-selection choices are not recorded as predictions. "
            "Bonferroni over 320 is therefore anti-conservative for the true search."
        ),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "paired_significance_wave27.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
