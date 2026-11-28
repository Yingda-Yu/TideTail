"""Read-only Issue #5 audit; never infer storm identity from window order."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(predictions, source):
    files = sorted(predictions.glob("*.npz"))
    records = []
    regions = {}
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta_json"]))
            pred = data["pred_residual"]
            obs = data["obs_residual"]
            records.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "meta": meta,
                    "shape": list(pred.shape),
                    "residual_mae_m": float(np.abs(pred - obs).mean()),
                    "identity_fields": sorted(
                        set(data.files) & {"event_ids", "station_ids", "timestamps", "window_ids"}
                    ),
                }
            )
            region = meta["region"]
            if region in regions:
                continue
            base = source / region
            with np.load(base / "window_index_v1.npz") as wi:
                starts = wi["test_start_indices"]
                context, horizon = int(wi["input_len"]), int(wi["horizon"])
                target = pd.read_csv(base / "target_timeline_v1.csv")
                tide = pd.read_csv(base / "tide_timeline_v1.csv")
                merged = target.merge(
                    tide[["timestamp_utc", "tide_level_m"]],
                    on="timestamp_utc",
                    validate="one_to_one",
                )
                expected = merged.water_level_m.to_numpy()[
                    starts[:, None] + context + np.arange(horizon)
                ]
                events = pd.read_csv(base / "event_table_v1.csv")
                regions[region] = {
                    "n_test_windows": len(starts),
                    "horizon": horizon,
                    "adjacent_overlapping_target_pairs": int((np.diff(starts) < horizon).sum()),
                    "source_row_alignment_exact": bool(
                        np.array_equal(expected.astype(obs.dtype), obs)
                    ),
                    "reconstruction_max_abs_m": float(
                        np.abs(
                            merged.water_level_m
                            + merged.tide_level_m
                            - merged.observed_total_water_level_m
                        ).max()
                    ),
                    "station_ids": sorted(target.anchor_id.unique().tolist()),
                    "threshold_excursions_test": int((events["split"] == "test").sum()),
                    "event_definition": events.threshold_definition.unique().tolist(),
                    "canonical_storm_count": None,
                    "independent_storm_ci": None,
                    "reason": "No verified prediction-to-canonical-storm mapping; threshold excursions are not independent storms.",
                    "source_hashes": {
                        name: sha256(base / name)
                        for name in [
                            "window_index_v1.npz",
                            "event_table_v1.csv",
                            "region_manifest_v1.json",
                            "train_only_metadata_v1.json",
                        ]
                    },
                }
    lookup = {
        (r["meta"]["region"], r["meta"]["group"], r["meta"].get("model"), r["meta"].get("seed")): r
        for r in records
    }
    comparisons = []
    for r in records:
        m = r["meta"]
        if not m["group"].startswith("A3_"):
            continue
        baseline = lookup[(m["region"], "A0_real_only", m["model"], m["seed"])]
        comparisons.append(
            {
                "region": m["region"],
                "group": m["group"],
                "model": m["model"],
                "seed": m["seed"],
                "delta_mae_m": r["residual_mae_m"] - baseline["residual_mae_m"],
                "interpretation": "Descriptive window-weighted effect only; no storm CI.",
            }
        )
    return {
        "issue": 5,
        "source_sha": "86169e1cd219db11e0c9befe7eb0b0bf659e6f72",
        "n_prediction_files": len(files),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "regions": regions,
        "predictions": records,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.predictions, args.source), indent=2) + "\n")
