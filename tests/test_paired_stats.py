"""Regression tests for paired event-level significance helpers.

Locks the bootstrap p-value fix: the Monte-Carlo denominator must use a
count of hits, not a proportion, and a CI that includes zero must not
produce a tiny p-value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from paired_significance_test import paired_bootstrap, paired_randomization  # noqa: E402


def test_bootstrap_p_consistent_with_ci():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.2, 1.0, 15)
    boot = paired_bootstrap(diffs, 4000, 42)
    perm = paired_randomization(diffs, 4000, 42)
    assert boot["ci95_low"] < 0 < boot["ci95_high"]
    assert boot["ci_excludes_zero"] is False
    assert boot["p_value"] > 0.05
    assert perm["p_value"] > 0.05


def test_bootstrap_detects_clear_effect():
    rng = np.random.default_rng(1)
    diffs = rng.normal(-1.0, 0.2, 20)
    boot = paired_bootstrap(diffs, 4000, 7)
    assert boot["ci95_high"] < 0
    assert boot["ci_excludes_zero"] is True
    assert boot["p_value"] < 0.01


def test_bootstrap_p_never_zero():
    diffs = np.array([-1.0] * 10)
    boot = paired_bootstrap(diffs, 2000, 3)
    assert boot["p_value"] > 0
    assert boot["p_value"] <= 1.0
