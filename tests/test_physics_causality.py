"""Causality tests for physics-derived Flow Matching features.

These lock the contract that ERA5 covariates summarise ONLY the 96 h
conditioning history [s, s+96) and never the prediction horizon.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "wave27_physics_fm", ROOT / "scripts" / "wave27_physics_fm.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["wave27_physics_fm"] = mod
spec.loader.exec_module(mod)

CONTEXT = 96
N = 2000


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "msl": 101000.0 + np.cumsum(rng.normal(0, 5, N)),
        "u10": rng.normal(0, 3, N),
        "v10": rng.normal(0, 3, N),
        "speed": np.abs(rng.normal(0, 3, N)),
    })


def test_features_ignore_future_horizon():
    """Mutating the horizon must not change features for that window.

    Each start is tested in isolation because mutating forward from one
    window's origin can legitimately alter a later window's history.
    """
    base = _frame()
    for s in (500, 900):
        starts = np.array([s])
        perturbed = base.copy()
        perturbed.iloc[s + CONTEXT:] = 1.0e6
        a, stats = mod.build_features(starts, base, "combined")
        b, _ = mod.build_features(starts, perturbed, "combined", stats)
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_features_react_to_history_values():
    starts = np.array([500])
    base = _frame()
    perturbed = base.copy()
    perturbed.iloc[starts[0]:starts[0] + CONTEXT, :] += 10.0
    a, stats = mod.build_features(starts, base, "pressure")
    b, _ = mod.build_features(starts, perturbed, "pressure", stats)
    assert not np.allclose(a, b)


def test_no_empty_window_at_origin():
    starts = np.array([0])
    feats, _ = mod.build_features(starts, _frame(), "combined")
    assert feats.shape == (1, 9)
    assert np.all(np.isfinite(feats))


def test_standard_has_zero_physics_dimension():
    feats, _ = mod.build_features(np.array([10]), _frame(), "standard")
    assert feats.shape == (1, 0)


# ---------------------------------------------------------------------------
# E26 sampling-path regression test
# ---------------------------------------------------------------------------

class _IdentityFM:
    """Stub generator returning its (normalised) history input as the
    sample: if sample_windows normalises/de-normalises correctly the
    round trip reproduces the raw input exactly (E26 fix)."""

    def sample(self, hn, net, seed=0, physics=None):
        return hn


def test_sample_windows_normalises_history_and_de_normalises_output():
    rng = np.random.default_rng(3)
    hist = (rng.normal(0, 0.2, (6, CONTEXT)) + 0.35).astype(np.float32)
    model = {"net": None, "fm": _IdentityFM(), "physics_dim": 0}
    out = mod.sample_windows(model, hist, np.zeros((6, 0)),
                             np.arange(6), draws=1)
    np.testing.assert_allclose(out[0], hist, atol=1e-5)
