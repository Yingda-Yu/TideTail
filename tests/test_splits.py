"""Event-grouped chronological split tests."""
import numpy as np

from tidetail.splits import (
    EventGroup,
    group_by_boolean_flags,
    indices_from_groups,
    partition_chronological,
)


def test_group_by_flags_merges_close_runs():
    flags = np.zeros(100, dtype=bool)
    flags[10:15] = True   # run A
    flags[17:20] = True   # run B — 1 index away from A, should merge
    flags[70:75] = True   # run C — far away
    runs = group_by_boolean_flags(flags, gap_tolerance=3)
    # A and B merge → (10, 19); C stays → (70, 74)
    assert len(runs) == 2
    assert runs[0] == (10, 19)
    assert runs[1] == (70, 74)


def test_partition_keeps_event_groups_atomic():
    groups = [EventGroup(f"event_{i}", start_idx=200 * i, end_idx=200 * i + 150)
              for i in range(12)]
    train, val, test = partition_chronological(groups, 0.6, 0.2)
    assert len(train) == 7 and len(val) == 2 and len(test) == 3
    # chronological
    assert train[-1].start_idx < val[0].start_idx
    assert val[-1].start_idx < test[0].start_idx
    # no shared events
    all_ids = {g.event_id for g in train} | {g.event_id for g in val} | {g.event_id for g in test}
    assert len(all_ids) == 12


def test_indices_from_groups_covers_full_range():
    groups = [EventGroup("a", 0, 5), EventGroup("b", 10, 12)]
    idx = indices_from_groups(groups)
    assert set(idx.tolist()) == {0, 1, 2, 3, 4, 5, 10, 11, 12}
