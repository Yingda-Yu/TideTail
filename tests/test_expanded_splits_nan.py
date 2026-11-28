"""Wave 3.0 regression: expanded splits must exclude NaN windows.

Locks the fix for the southeast_australia A0=NaN / thames NaN-training
failure: build_expanded_splits builds strict-containment windows from the
raw series, so without the NaN filter any missing water-level hour
contaminated up to 168 windows (history + target). The frozen primary
window_index_v1 protocol already excludes such windows; the expanded
splits must do the same.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


w27e = _load("wave27_expanded_eval",
             ROOT / "scripts" / "wave27_expanded_eval.py")


def _ts_index(n_hours: int, start="2015-01-01") -> pd.Series:
    ts = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    return pd.Series(ts)


def test_nan_windows_excluded_from_all_splits():
    n = 87672
    series = np.sin(np.arange(n) / 24.0).astype(np.float32)
    ts = _ts_index(n)
    # punch a 31-hour gap inside the test years (like southeast_australia)
    series[81000:81031] = np.nan
    spl = w27e.build_expanded_splits(ts, series)
    for split in ("train", "val", "test"):
        h, t = spl[f"{split}_h"], spl[f"{split}_t"]
        assert np.isfinite(h).all(), f"{split} history has NaN"
        assert np.isfinite(t).all(), f"{split} target has NaN"
        excluded = spl[f"{split}_nan_windows_excluded"]
        assert excluded >= 0
    # the gap lies in the test span: contaminated windows must be counted
    assert spl["test_nan_windows_excluded"] > 0
    # starts remain strictly contiguous-within-split after filtering
    st = spl["test_starts"]
    assert len(st) == len(spl["test_h"])


def test_clean_series_matches_unfiltered_counts():
    n = 87672
    series = np.sin(np.arange(n) / 24.0).astype(np.float32)
    ts = _ts_index(n)
    spl = w27e.build_expanded_splits(ts, series)
    # no NaN anywhere -> nothing excluded, counts identical to pre-fix
    assert all(spl[f"{k}_nan_windows_excluded"] == 0
               for k in ("train", "val", "test"))
    marks = [int(ts.searchsorted(pd.Timestamp(b, tz="UTC")))
             for b in w27e.BOUNDS]
    span = w27e.CONTEXT + w27e.HORIZON
    assert len(spl["train_h"]) == marks[1] - marks[0] - span + 1
