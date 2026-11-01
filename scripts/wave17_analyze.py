"""Wave 1.7 — Post-hoc analysis of all saved predictions.

Reads every .npz in data/predictions/v2/, computes total-water metrics,
runs event-level paired bootstrap CI, and produces a consolidated results
JSON + markdown summary.

Usage:
    python3 scripts/wave17_analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data" / "predictions" / "v2"
RESULTS_DIR = ROOT / "data" / "results"
OUT_JSON = RESULTS_DIR / "wave17_analysis_v2.json"
OUT_MD = RESULTS_DIR / "wave17_analysis_v2.md"

REGIONS = [
    "ny_bight", "thames_north_sea", "us_gulf_coast",
    "south_atlantic_charleston", "southeast_australia",
]


def load_prediction(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
    return {
        "pred_residual": d["pred_residual"].astype(np.float64),
        "obs_residual": d["obs_residual"].astype(np.float64),
        "future_tide": d["future_tide"].astype(np.float64),
        "meta": meta,
        "path": str(path),
        "filename": path.stem,
    }


def total_water_eval(pa: dict) -> dict:
    pred_water = pa["pred_residual"] + pa["future_tide"]
    obs_water = pa["obs_residual"] + pa["future_tide"]
    err = pred_water - obs_water
    return {
        "tw_mae": float(np.mean(np.abs(err))),
        "tw_rmse": float(np.sqrt(np.mean(err ** 2))),
        "tw_bias": float(np.mean(err)),
        "tw_corr": float(np.corrcoef(pred_water.reshape(-1), obs_water.reshape(-1))[0, 1]),
        "tw_peak_mag_mae": float(np.mean(np.abs(
            pred_water[np.arange(len(pred_water)), pred_water.argmax(1)] -
            obs_water[np.arange(len(obs_water)), obs_water.argmax(1)]
        ))),
        "tw_peak_timing_mae_hours": float(np.mean(np.abs(
            pred_water.argmax(1) - obs_water.argmax(1)
        ))),
    }


def paired_bootstrap(pa_a: dict, pa_b: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap on per-window MAE (event-wise resampling fallback)."""
    pred_a = pa_a["pred_residual"] + pa_a["future_tide"]
    pred_b = pa_b["pred_residual"] + pa_b["future_tide"]
    obs = pa_a["obs_residual"] + pa_a["future_tide"]
    assert np.allclose(obs, pa_b["obs_residual"] + pa_b["future_tide"]), "target mismatch"

    mae_a = np.abs(pred_a - obs).mean()
    mae_b = np.abs(pred_b - obs).mean()
    delta = mae_a - mae_b

    # Window-level bootstrap (event IDs not available in prediction files)
    rng = np.random.default_rng(seed)
    n = len(obs)
    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = np.abs(pred_a[idx] - obs[idx]).mean() - np.abs(pred_b[idx] - obs[idx]).mean()

    ci_lo, ci_hi = float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))
    return {
        "delta_mae_a_minus_b": float(delta),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "candidate_better": ci_hi < 0,
        "baseline_better": ci_lo > 0,
        "neutral": not (ci_lo > 0 or ci_hi < 0),
        "bootstrap_units": f"forecast_windows (n={n})",
    }


def main():
    # Load all predictions
    predictions = {}
    for path in sorted(PRED_DIR.glob("*.npz")):
        pa = load_prediction(path)
        key = (pa["meta"].get("region", ""), pa["meta"].get("group", ""),
               pa["meta"].get("model", ""), pa["meta"].get("seed", ""))
        predictions[key] = pa
    print(f"Loaded {len(predictions)} prediction files")

    # Compute metrics per file
    all_results = []
    for key, pa in predictions.items():
        region, group, model, seed = key
        tw = total_water_eval(pa)
        all_results.append({
            "region": region, "group": group, "model": model, "seed": seed,
            "path": pa["path"],
            **tw,
        })

    # Save JSON
    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"Saved {len(all_results)} metrics → {OUT_JSON}")

    # Aggregate by region × group (mean across seeds for GRU)
    agg = {}
    for r in all_results:
        if r["model"] in ("gru", "tcn") and r["seed"] != 0:  # skip persistence/seasonal (seed=0)
            k = (r["region"], r["group"], r["model"])
            if k not in agg:
                agg[k] = []
            agg[k].append(r)

    # Build summary
    summary_lines = []
    summary_lines.append("# Wave 1.7 Analysis — Total Water Forecasting\n")
    summary_lines.append(f"**Files analyzed**: {len(predictions)}")
    summary_lines.append(f"**PRED_DIR**: `{PRED_DIR}`\n")

    for region in REGIONS:
        summary_lines.append(f"\n## {region}\n")
        summary_lines.append("| Group | Model | tw_mae ± sd | tw_tail95 | peak_mag_mae | peak_timing_h |")
        summary_lines.append("|-------|-------|------------:|----------:|-------------:|--------------:|")

        # Persistence + seasonal (seed 0, one row each)
        for grp, model in [("persistence", "persistence"), ("seasonal_24h", "seasonal_24h")]:
            k = (region, grp, model, 0)
            if k in predictions:
                r = next(x for x in all_results if x["region"] == region and x["group"] == grp and x["model"] == model)
                summary_lines.append(f"| {grp} | {model} | {r['tw_mae']:.4f} | - | {r['tw_peak_mag_mae']:.4f} | {r['tw_peak_timing_mae_hours']:.1f} |")

        # A0/A1/A2/A3 groups (aggregate seeds 42+99)
        for grp in sorted({r[1] for r in agg if r[0] == region}):
            for model in ["gru", "tcn"]:
                rows = agg.get((region, grp, model), [])
                if not rows:
                    continue
                maes = [r["tw_mae"] for r in rows]
                tails = [r.get("tw_tail_mae_q95_pos_tail_mae", float("nan")) for r in rows]
                pks = [r["tw_peak_mag_mae"] for r in rows]
                ptt = [r["tw_peak_timing_mae_hours"] for r in rows]
                summary_lines.append(
                    f"| {grp} | {model} | {np.mean(maes):.4f} ± {np.std(maes):.4f} | "
                    f"{np.nanmean(tails):.4f} | {np.mean(pks):.4f} | {np.mean(ptt):.1f} |"
                )

    # A3 vs A0 bootstrap CI (GRU, region-wise)
    summary_lines.append("\n## A3 vs A0 — Paired Bootstrap CI (tw_mae, window-level)\n")
    summary_lines.append("Delta = A3 - A0 (negative = A3 better)\n")
    summary_lines.append("| Region | A3_ratio | A3_seed | delta | CI_lo | CI_hi | verdict |")
    summary_lines.append("|--------|----------|---------|------:|------:|------:|:--------|")

    a3_results = []
    for key, pa_a3 in predictions.items():
        region, grp, model, seed = key
        if model != "gru" or not grp.startswith("A3_"):
            continue
        # Find matching A0: same region, model, seed
        a0_key = (region, "A0_real_only", "gru", seed)
        if a0_key not in predictions:
            continue
        pa_a0 = predictions[a0_key]
        ci = paired_bootstrap(pa_a3, pa_a0)
        a3_results.append({
            "region": region, "group_a3": grp, "seed": seed,
            "a3_mae": (pa_a3["pred_residual"] + pa_a3["future_tide"] - pa_a3["obs_residual"] - pa_a3["future_tide"]).__class__,
            "delta": ci["delta_mae_a_minus_b"],
            "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "verdict": "A3_better" if ci["candidate_better"] else ("A0_better" if ci["baseline_better"] else "neutral"),
        })
        ratio = grp.split("_")[-1] if "ratio_" in grp else ""
        fm_seed = grp.split("_s")[1].split("_")[0] if "_s" in grp else ""
        summary_lines.append(f"| {region} | {ratio} | s{fm_seed} | {ci['delta_mae_a_minus_b']:.4f} | "
                             f"{ci['ci_lo']:.4f} | {ci['ci_hi']:.4f} | {ci['verdict']} |")

    # A2 vs A0 bootstrap
    summary_lines.append("\n## A2 (history noise) vs A0 — Paired Bootstrap CI (tw_mae, GRU)\n")
    summary_lines.append("| Region | delta | CI_lo | CI_hi | verdict |")
    summary_lines.append("|--------|------:|------:|------:|:--------|")
    for region in REGIONS:
        for seed in [42, 99]:
            a2_key = (region, "A2_history_noise", "gru", seed)
            a0_key = (region, "A0_real_only", "gru", seed)
            if a2_key not in predictions or a0_key not in predictions:
                continue
            ci = paired_bootstrap(predictions[a2_key], predictions[a0_key])
            summary_lines.append(f"| {region} (seed{seed}) | {ci['delta_mae_a_minus_b']:.4f} | "
                                 f"{ci['ci_lo']:.4f} | {ci['ci_hi']:.4f} | {ci['verdict']} |")

    # A1 vs A0
    summary_lines.append("\n## A1 (oversampling) vs A0 — Paired Bootstrap CI (tw_mae, GRU)\n")
    summary_lines.append("| Region | delta | CI_lo | CI_hi | verdict |")
    summary_lines.append("|--------|------:|------:|------:|:--------|")
    for region in REGIONS:
        for seed in [42, 99]:
            a1_key = (region, "A1_oversampling", "gru", seed)
            a0_key = (region, "A0_real_only", "gru", seed)
            if a1_key not in predictions or a0_key not in predictions:
                continue
            ci = paired_bootstrap(predictions[a1_key], predictions[a0_key])
            summary_lines.append(f"| {region} (seed{seed}) | {ci['delta_mae_a_minus_b']:.4f} | "
                                 f"{ci['ci_lo']:.4f} | {ci['ci_hi']:.4f} | {ci['verdict']} |")

    OUT_MD.write_text("\n".join(summary_lines))
    print(f"\nSaved markdown summary → {OUT_MD}")
    print("\nSummary preview:")
    print("\n".join(summary_lines[:40]))


if __name__ == "__main__":
    main()
