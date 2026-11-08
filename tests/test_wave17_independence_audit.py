"""An empty archive cannot establish event counts or significance."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).parents[1] / "scripts/wave17_independence_audit.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empty_archive_has_no_claims(tmp_path):
    result = module.audit(tmp_path, tmp_path)
    assert result["n_prediction_files"] == 0
    assert result["regions"] == {}
    assert result["comparisons"] == []


def test_overlapping_excursions_never_become_storms(tmp_path):
    import json

    import numpy as np
    import pandas as pd

    source = tmp_path / "region"
    source.mkdir()
    pd.DataFrame(
        {
            "timestamp_utc": range(5),
            "water_level_m": range(5),
            "observed_total_water_level_m": range(5),
            "anchor_id": ["station"] * 5,
        }
    ).to_csv(source / "target_timeline_v1.csv", index=False)
    pd.DataFrame({"timestamp_utc": range(5), "tide_level_m": [0] * 5}).to_csv(
        source / "tide_timeline_v1.csv", index=False
    )
    pd.DataFrame({"split": ["test", "test"], "threshold_definition": ["q95", "q95"]}).to_csv(
        source / "event_table_v1.csv", index=False
    )
    for name in ["region_manifest_v1.json", "train_only_metadata_v1.json"]:
        (source / name).write_text("{}")
    np.savez(source / "window_index_v1.npz", test_start_indices=[0, 1], input_len=1, horizon=2)
    obs = np.array([[1, 2], [2, 3]])
    for group, shift in [("A0_real_only", 1), ("A3_test", 0)]:
        np.savez(
            tmp_path / f"{group}.npz",
            pred_residual=obs + shift,
            obs_residual=obs,
            future_tide=np.zeros_like(obs),
            meta_json=json.dumps({"region": "region", "group": group, "model": "gru", "seed": 42}),
        )
    result = module.audit(tmp_path, tmp_path)
    region = result["regions"]["region"]
    assert region["adjacent_overlapping_target_pairs"] == 1
    assert region["threshold_excursions_test"] == 2
    assert region["canonical_storm_count"] is None
    assert region["independent_storm_ci"] is None
    assert region["source_row_alignment_exact"]
    assert result["comparisons"][0]["delta_mae_m"] == -1
