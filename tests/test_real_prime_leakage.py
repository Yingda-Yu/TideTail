"""Verify leakage-zero invariants on the migrated Prime dataset.

This test runs only when the dataset is present at /data/yingda/tidetail-data/.
Skipped automatically on CI machines without the data copy.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
HAS_DATA = DATA_ROOT.is_dir()
REGIONS = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()]) if HAS_DATA else []

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="Prime dataset not migrated")


def _f(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


class TestRealPrimeData:
    @pytest.mark.parametrize("region", REGIONS)
    def test_chronological_splits(self, region):
        base = DATA_ROOT / region
        with open(base / "train_only_metadata_v1.json") as fh:
            meta = json.load(fh)
        with open(base / "target_timeline_v1.csv") as fh:
            rows = list(csv.DictReader(fh))
        train_end = int(meta["train_end_index"])
        val_end = int(meta["validation_end_index"])
        assert train_end < val_end < len(rows)
        assert rows[train_end]["timestamp_utc"] < rows[train_end + 1]["timestamp_utc"]
        assert rows[val_end]["timestamp_utc"] < rows[val_end + 1]["timestamp_utc"]

    @pytest.mark.parametrize("region", REGIONS)
    def test_train_only_thresholds(self, region):
        base = DATA_ROOT / region
        with open(base / "train_only_metadata_v1.json") as fh:
            meta = json.load(fh)
        with open(base / "target_timeline_v1.csv") as fh:
            rows = list(csv.DictReader(fh))
        train_end = int(meta["train_end_index"])
        all_obs = np.array([_f(r.get("observed_total_water_level_m", "")) for r in rows])
        train_q95 = float(np.nanquantile(all_obs[:train_end], 0.95))
        train_q99 = float(np.nanquantile(all_obs[:train_end], 0.99))
        assert abs(train_q95 - meta["thresholds"]["total_water_q95_m"]) < 0.01
        assert abs(train_q99 - meta["thresholds"]["total_water_q99_m"]) < 0.01

    @pytest.mark.parametrize("region", REGIONS)
    def test_disjoint_event_groups(self, region):
        base = DATA_ROOT / region
        with open(base / "event_table_v1.csv") as fh:
            events = list(csv.DictReader(fh))
        t = {e["event_id"] for e in events if e["split"] == "train"}
        v = {e["event_id"] for e in events if e["split"] == "validation"}
        te = {e["event_id"] for e in events if e["split"] == "test"}
        assert not (t & v) and not (t & te) and not (v & te)
        assert t and v and te  # every partition populated


class TestRealPrimeBaselines:
    @pytest.mark.parametrize("region", REGIONS)
    def test_seasonal_beats_persistence(self, region):
        """Seasonal MAE should beat raw persistence on real tide data."""
        from tidetail.baselines import persistence, seasonal_baseline
        from tidetail.evaluate import mae

        base = DATA_ROOT / region
        with open(base / "train_only_metadata_v1.json") as fh:
            meta = json.load(fh)
        with open(base / "target_timeline_v1.csv") as fh:
            rows = list(csv.DictReader(fh))
        all_obs = np.array([_f(r.get("observed_total_water_level_m", "")) for r in rows])
        test = all_obs[int(meta["validation_end_index"]):]
        horizon, input_len = 24, 72
        windows = []
        for s in range(len(test) - horizon - input_len + 1):
            x = test[s:s + input_len]; y = test[s + input_len:s + input_len + horizon]
            if np.isfinite(x).all() and np.isfinite(y).all():
                windows.append((x, y))
        assert len(windows) >= 100, f"{region}: too-few windows {len(windows)}"
        pers_mae = np.mean([mae(y, persistence(x, horizon)) for x, y in windows])
        seas_mae = np.mean([mae(y, seasonal_baseline(x, horizon, period=24)) for x, y in windows])
        assert seas_mae < pers_mae, f"{region}: seasonal ({seas_mae:.4f}) did not beat persistence ({pers_mae:.4f})"
