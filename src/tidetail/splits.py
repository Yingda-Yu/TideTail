"""Leakage-safe data partitioning.

The protocol enforces three hard gates:
  1. Event-group disjointness — no event appears in two partitions.
  2. Chronological ordering — train < val < test end times.
  3. Thresholds from train only — q95/q99 are never derived from val/test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import GaugeRecord


@dataclass
class EventGroup:
    """One independent storm or extreme event — must stay atomic across splits."""

    event_id: str
    start_idx: int
    end_idx: int          # inclusive
    label: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.end_idx - self.start_idx + 1


def group_by_boolean_flags(flags: np.ndarray, gap_tolerance: int = 1) -> List[Tuple[int, int]]:
    """Return (start_idx, end_idx_inclusive) runs where flags==True.

    Runs whose False-valued gap is <= gap_tolerance are merged.
    gap_tolerance=1 means runs separated by a single False index are combined.
    """
    flags = np.asarray(flags, dtype=bool)
    n = flags.size
    runs: List[Tuple[int, int]] = []
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
                            min_events_per_split: int = 1) -> Tuple[List[EventGroup], List[EventGroup], List[EventGroup]]:
    """Split events by chronological start, ensuring each partition has ≥min_events_per_split.

    If the requested fracs would leave any partition below min_events_per_split,
    we shrink the smaller partitions until constraints hold.
    """
    if not groups:
        raise ValueError("No event groups to split")
    events = sorted(groups, key=lambda g: g.start_idx)
    n = len(events)
    n_train = max(min_events_per_split, int(round(n * train_frac)))
    n_val = max(min_events_per_split, int(round(n * val_frac)))
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


def train_only_thresholds(train: np.ndarray, quantiles: Sequence[float] = (0.95, 0.99)) -> Dict[float, float]:
    """Compute extreme-event thresholds from TRAIN ONLY.

    Returns {q: value} for each requested quantile.
    """
    finite = np.isfinite(train)
    if finite.sum() < 10:
        raise ValueError("Too few train samples for threshold estimation")
    return {q: float(np.nanquantile(train, q)) for q in quantiles}


def indices_from_groups(groups: Sequence[EventGroup]) -> np.ndarray:
    """Return flat integer indices covered by all event groups."""
    idx: List[int] = []
    for g in groups:
        idx.extend(range(g.start_idx, g.end_idx + 1))
    return np.array(idx, dtype=int)


def embargo_filter(time_idx: np.ndarray, embargo: int) -> Callable[[EventGroup], bool]:
    """Return predicate True for groups whose indices aren't within `embargo` of `time_idx`."""
    def _ok(g: EventGroup) -> bool:
        neighbors = np.abs(time_idx - np.mean([g.start_idx, g.end_idx]))
        return (neighbors.min() > embargo) if neighbors.size else True
    return _ok
