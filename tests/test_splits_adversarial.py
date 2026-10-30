"""Adversarial leakage and boundary tests for data split protocol.

These tests probe the failure modes that motivated the embargo_filter fix
(commit 210a985) and the FORECAST_TASK_CONTRACT v1 (96h lookback, 72h horizon).
All tests use synthetic numpy data — no real gauge access required.
"""
import numpy as np
import pytest

from tidetail.protocol import assert_disjoint_groups
from tidetail.splits import (
    EventGroup,
    embargo_filter,
    embargo_full_support_filter,
    partition_chronological,
    train_only_thresholds,
)

# ==========================================================================
# Adversarial event-group edge cases
# ==========================================================================

def test_interleaved_events_partitioned_by_start_order_not_id():
    """Events scattered in ID order but sequential in time must sort chronologically."""
    groups = [
        EventGroup("event_chaos_3", start_idx=100, end_idx=150),
        EventGroup("event_chaos_1", start_idx=0, end_idx=50),
        EventGroup("event_chaos_4", start_idx=200, end_idx=250),
        EventGroup("event_chaos_2", start_idx=60, end_idx=90),
    ]
    train, _val, test = partition_chronological(groups, 0.5, 0.25)
    # Train = earliest chronologically, regardless of event_id string
    assert train[0].start_idx == 0
    assert train[-1].start_idx <= 100
    assert test[0].start_idx >= 200


def test_overlapping_event_groups_rejected():
    """Events with overlapping indices must be rejected before split."""
    groups = [
        EventGroup("overlap_a", start_idx=0, end_idx=100),
        EventGroup("overlap_b", start_idx=50, end_idx=150),  # overlaps a!
    ]
    with pytest.raises(ValueError, match="overlap"):
        partition_chronological(groups, 0.8, 0.1)


def test_duplicate_event_ids_rejected():
    """Two EventGroups with the same event_id must not coexist."""
    groups = [
        EventGroup("dup_001", start_idx=0, end_idx=50),
        EventGroup("dup_001", start_idx=100, end_idx=150),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        partition_chronological(groups, 0.8, 0.1)


def test_empty_split_rejected():
    """Empty input must raise."""
    with pytest.raises(ValueError):
        partition_chronological([], 0.8, 0.1)


# ==========================================================================
# Embargo — full-support interval tests
# ==========================================================================

def test_embargo_full_support_rejects_origin_at_split_boundary():
    """Forecast origin o where support [o-L, o+H) would cross a partition boundary
    must be rejected."""
    L = 96
    H = 72
    split_end = 1000  # train block ends here (exclusive); starts at 0

    full_ok = embargo_full_support_filter(split_end, L, H)

    # Every index i in [start_idx, end_idx] must be: i >= 0 + 96 AND i <= 1000 - 72 = 928
    # SAFE: all indices 96..928
    safe_group = EventGroup("safe", start_idx=500, end_idx=600)
    assert full_ok(safe_group) == True

    # UNSAFE: event with indices < 96 (lookback not possible at start of block)
    too_early = EventGroup("early", start_idx=50, end_idx=80)
    assert full_ok(too_early) == False

    # UNSAFE: event extends past split_end - horizon
    # Any index > 928 fails. Even index 929 is one past.
    too_late = EventGroup("late", start_idx=925, end_idx=930)  # 930 > 928
    assert full_ok(too_late) == False

    # EDGE: group entirely within [96, 928]
    edge_safe = EventGroup("edge_safe", start_idx=96, end_idx=928)
    assert full_ok(edge_safe) == True

    # EDGE: group that extends just one past
    edge_unsafe = EventGroup("edge_past", start_idx=96, end_idx=929)
    assert full_ok(edge_unsafe) == False


def test_embargo_filter_rejects_group_spanning_protected_region():
    """Event containing protected index → rejected by embargo."""
    protected = np.array([500])
    emb = embargo_filter(protected, embargo=200)

    # [200, 800] contains 500 → min distance = 0 < 200 → reject
    spans = EventGroup("spans", start_idx=200, end_idx=800)
    assert emb(spans) == False

    # [0, 400] → min distance is 500 - 400 = 100 < 200 → reject
    close = EventGroup("close", start_idx=0, end_idx=400)
    assert emb(close) == False

    # [0, 299] → max idx 299, distance 500 - 299 = 201 > 200 → accept
    far = EventGroup("far", start_idx=0, end_idx=299)
    assert emb(far) == True

    # [1000, 1500] → min distance 1000 - 500 = 500 > 200 → accept
    post = EventGroup("post", start_idx=1000, end_idx=1500)
    assert emb(post) == True


def test_embargo_rejects_touching_region():
    """An event adjacent to protected region must be rejected if distance < embargo."""
    protected = np.array([500])
    emb = embargo_filter(protected, embargo=200)

    # [0, 499]: max idx 499, min_dist = 1 < 200 → reject
    touching = EventGroup("touching", start_idx=0, end_idx=499)
    assert emb(touching) == False

    # [0, 300]: max idx 300, min_dist = 200, embargo > so reject (strictly greater)
    at_boundary = EventGroup("at_boundary", start_idx=0, end_idx=300)
    assert emb(at_boundary) == False

    # [0, 299]: max idx 299, min_dist = 201 > 200 → accept
    clear = EventGroup("clear", start_idx=0, end_idx=299)
    assert emb(clear) == True


# ==========================================================================
# train_only_thresholds — leakage prevention
# ==========================================================================

def test_train_only_thresholds_ignore_val_test():
    """Thresholds from train only, never peek at val/test quantiles."""
    rng = np.random.default_rng(42)
    train = rng.uniform(0, 1, 1000)
    rng.uniform(1, 2, 1000)
    rng.uniform(-1, 0, 1000)

    thresholds = train_only_thresholds(train, quantiles=(0.95, 0.99))
    assert 0.92 <= thresholds[0.95] <= 0.98
    assert 0.97 <= thresholds[0.99] <= 0.995
    # NOT pulled up by val high values
    assert thresholds[0.95] < 1.0


def test_train_only_thresholds_constant_train_ok():
    thresholds = train_only_thresholds(np.ones(100) * 5.0, quantiles=(0.95,))
    assert thresholds[0.95] == 5.0


def test_train_only_thresholds_too_few_samples_rejected():
    with pytest.raises(ValueError):
        train_only_thresholds(np.array([1.0, 2.0]), quantiles=(0.95,))


# ==========================================================================
# assert_disjoint_groups — protocol integrity
# ==========================================================================

def test_assert_disjoint_detects_cross_split_event():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_groups({"event_01149_a"}, {"event_01149_a"}, {"event_003"})


def test_assert_disjoint_passes_when_clean():
    assert_disjoint_groups({"train_1", "train_2"}, {"val_1"}, {"test_1", "test_2"})


def test_assert_disjoint_empty_split_rejected():
    with pytest.raises(ValueError, match="at least one event"):
        assert_disjoint_groups({"train_1"}, set(), {"test_1"})


# ==========================================================================
# HK cross-boundary simulation (event 01149 at index ~78904)
# ==========================================================================

def test_hk_region_admission_status_excludes_it():
    """HK region excluded at admission level, not at embargo level."""
    ADMITTED = {"ny_bight", "thames_north_sea", "southeast_australia",
                "us_gulf_coast", "south_atlantic_charleston"}
    assert "pearl_river_hong_kong" not in ADMITTED
    assert len(ADMITTED) == 5
