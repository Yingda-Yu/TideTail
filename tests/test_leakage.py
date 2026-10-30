"""Regression tests — hard leakage gates.

These MUST catch any accidental cross-partition contamination.
"""
import numpy as np
import pytest

from tidetail.protocol import assert_disjoint_groups
from tidetail.splits import (
    EventGroup,
    partition_chronological,
    train_only_thresholds,
)

# ── Test 1: Thresholds from train only ────────────────────────────────

def test_thresholds_ignore_val_and_test():
    """Extreme thresholds must be train-only, never computed on val/test."""
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=1000)
    # val/test have extreme outliers that must NOT influence thresholds
    val = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 0.1, 20)])
    test = np.concatenate([rng.normal(0, 1, 100), rng.normal(20, 0.1, 10)])

    thresh = train_only_thresholds(train, quantiles=(0.95, 0.99))
    # Same call on val/test would catch different tails; ours MUST NOT
    thresh_val = train_only_thresholds(val, quantiles=(0.95, 0.99))
    thresh_test = train_only_thresholds(test, quantiles=(0.95, 0.99))

    # Train thresholds should be lower (no injected extremes)
    assert thresh[0.95] < thresh_val[0.95]
    assert thresh[0.95] < thresh_test[0.95]
    assert thresh[0.99] < thresh_val[0.99]
    assert thresh[0.99] < thresh_test[0.99]


# ── Test 2: Event groups stay atomic ─────────────────────────────────

def test_event_groups_never_split():
    """One event must land entirely in one partition or the other."""
    groups = [EventGroup(f"e{i}", start_idx=100 * i, end_idx=100 * i + 50)
              for i in range(12)]
    train, val, test = partition_chronological(groups, 0.6, 0.2, min_events_per_split=1)
    all_train = [g.event_id for g in train]
    all_val = [g.event_id for g in val]
    all_test = [g.event_id for g in test]
    # no overlap
    assert not (set(all_train) & set(all_val))
    assert not (set(all_train) & set(all_test))
    assert not (set(all_val) & set(all_test))


def test_chronological_order_of_partitions():
    """Train < val < test by start_idx must hold."""
    groups = [EventGroup(f"e{i}", start_idx=100 * i, end_idx=100 * i + 50)
              for i in range(12)]
    train, val, test = partition_chronological(groups, 0.6, 0.2)
    assert train[-1].start_idx < val[0].start_idx
    assert val[-1].start_idx < test[0].start_idx


# ── Test 3: Multi-station cross-contamination guard ──────────────────

def test_disjoint_groups_protocol_enforces_all_three():
    """assert_disjoint_groups must reject any pair."""
    # clean
    assert_disjoint_groups(["e1", "e2"], ["e3"], ["e4"])
    # any pair shared → ValueError
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_groups(["e1"], ["e1"], ["e2"])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_groups(["e1"], ["e2"], ["e1"])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_groups(["e1"], ["e2"], ["e2"])


# ── Test 4: Minimum partition size ───────────────────────────────────

def test_partitions_get_at_least_one_each():
    groups = [EventGroup(f"e{i}", start_idx=i * 100, end_idx=i * 100 + 50)
              for i in range(3)]
    train, val, test = partition_chronological(groups, 0.6, 0.2, min_events_per_split=1)
    assert len(train) >= 1 and len(val) >= 1 and len(test) >= 1
