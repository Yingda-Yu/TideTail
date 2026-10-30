"""Leakage-safe data partitioning.

The protocol enforces three hard gates:
  1. Event-group disjointness — no event appears in two partitions.
  2. Chronological ordering — train < val < test end times.
  3. Thresholds from train only — q95/q99 are never derived from val/test.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventGroup:
    """One independent storm or extreme event — must stay atomic across splits."""

    event_id: str
    start_idx: int
    end_idx: int          # inclusive
    label: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.end_idx - self.start_idx + 1


def group_by_boolean_flags(flags: np.ndarray, gap_tolerance: int = 1) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_inclusive) runs where flags==True.

    Runs whose False-valued gap is <= gap_tolerance are merged.
    gap_tolerance=1 means runs separated by a single False index are combined.
    """
    flags = np.asarray(flags, dtype=bool)
    n = flags.size
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            # Now j = one past end of current run.
            # Look ahead: if next gap is <= gap_tolerance and flags resume, merge.
            k = j
            while k < n and not flags[k] and (k - j) <= gap_tolerance:
                k += 1
            if k < n and flags[k] and (k - j) <= gap_tolerance:
                # merge: extend to end of next run
                while k < n and flags[k]:
                    k += 1
                runs.append((i, k - 1))
                i = k
            else:
                runs.append((i, j - 1))
                i = j
        else:
            i += 1
    return runs


def partition_chronological(groups: Sequence[EventGroup],
                            train_frac: float, val_frac: float,
                            min_events_per_split: int = 1) -> tuple[list[EventGroup], list[EventGroup], list[EventGroup]]:
    """Split events by chronological start, ensuring each partition has ≥min_events_per_split.

    If the requested fracs would leave any partition below min_events_per_split,
    we shrink the smaller partitions until constraints hold.
    """
    if not groups:
        raise ValueError("No event groups to split")
    events = sorted(groups, key=lambda g: g.start_idx)
    seen = set()
    previous_end = -1
    for event in events:
        if event.start_idx < 0 or event.end_idx < event.start_idx:
            raise ValueError("Invalid event interval")
        if event.event_id in seen:
            raise ValueError("Duplicate event ID; consolidate the event before splitting")
        if event.start_idx <= previous_end:
            raise ValueError("Event intervals overlap; consolidate events before splitting")
        seen.add(event.event_id)
        previous_end = event.end_idx
    n = len(events)
    n_train = max(min_events_per_split, round(n * train_frac))
    n_val = max(min_events_per_split, round(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < min_events_per_split:
        deficit = min_events_per_split - n_test
        n_test = min_events_per_split
        n_val = max(min_events_per_split, n_val - deficit)
    if n_val < min_events_per_split:
        deficit = min_events_per_split - n_val
        n_val = min_events_per_split
        n_train = max(min_events_per_split, n_train - deficit)
    # Final safety: n_train may now be too big → trim
    while n_train + n_val + n_test > n and n_train > min_events_per_split:
        n_train -= 1
    while n_train + n_val + n_test < n:
        # add remainder to test
        n_test += 1
    assert n_train + n_val + n_test == n, f"{n_train}+{n_val}+{n_test} != {n}"
    train = events[:n_train]
    val = events[n_train:n_train + n_val]
    test = events[n_train + n_val:]
    return train, val, test


def train_only_thresholds(train: np.ndarray, quantiles: Sequence[float] = (0.95, 0.99)) -> dict[float, float]:
    """Compute extreme-event thresholds from TRAIN ONLY.

    Returns {q: value} for each requested quantile.
    """
    finite = np.isfinite(train)
    if finite.sum() < 10:
        raise ValueError("Too few train samples for threshold estimation")
    return {q: float(np.nanquantile(train, q)) for q in quantiles}


def indices_from_groups(groups: Sequence[EventGroup]) -> np.ndarray:
    """Return flat integer indices covered by all event groups."""
    idx: list[int] = []
    for g in groups:
        idx.extend(range(g.start_idx, g.end_idx + 1))
    return np.array(idx, dtype=int)


def embargo_filter(time_idx: np.ndarray, embargo: int) -> Callable[[EventGroup], bool]:
    """Return predicate True for groups whose full index interval isn't within `embargo` of `time_idx`.

    Correct implementation: check every index [start_idx, end_idx] against every
    protected time_idx. The OLD midpoint-only version was unsafe — see PR #19 audit
    report "Embargo FAIL as general guarantee" and PR #21 recovery report §B.

    For a forecast with lookback L and horizon H, the full support of origin time o
    is [o-L, o+H). Two partitions' support intervals must NOT overlap.
    """
    time_idx = np.asarray(time_idx, dtype=int)
    if time_idx.size == 0:
        return lambda _g: True
    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    def _ok(g: EventGroup) -> bool:
        # All indices covered by this event group
        group_indices = np.arange(g.start_idx, g.end_idx + 1)
        if group_indices.size == 0:
            return True
        # Distance matrix: group_indices × time_idx
        dists = np.abs(group_indices[:, None] - time_idx[None, :])
        min_dist = dists.min()
        return min_dist > embargo

    return _ok


def embargo_full_support_filter(
    split_end: int, lookback: int, horizon: int
) -> Callable[[EventGroup], bool]:
    """Reject forecast origins whose full support [o-L, o+H) would cross split_end.

    For an hourly forecast window with lookback L and horizon H, origin time o has
    full support interval [o-L, o+H). If partition row block is [a, split_end),
    only origins with o >= a+L AND o <= split_end-H are safe. Any origin whose
    support overlaps another partition's row block is a leakage risk.

    Args:
        split_end: row index at which this partition ends (exclusive).
        lookback: historical window length (L).
        horizon: forecast horizon (H).

    Returns:
        Predicate True for EventGroups whose every index in [start_idx, end_idx] is
        a safe forecast origin time (i.e., o >= lookback AND o <= split_end - horizon).
    """
    if split_end <= 0 or lookback <= 0 or horizon <= 0:
        raise ValueError("split_end, lookback, horizon must be positive")

    def _ok(g: EventGroup) -> bool:
        safe_origin = (g.start_idx >= lookback) and (g.end_idx <= split_end - horizon)
        return safe_origin

    return _ok
