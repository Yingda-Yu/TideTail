"""CPU-only observation utilities. Split raw records before any preprocessing."""

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from numbers import Real

import numpy as np

from tidetail.protocol import assert_disjoint_groups


@dataclass(frozen=True)
class ObservationRecord:
    """One UTC gauge observation in meters; event IDs are global across stations.

    None/NaN levels are permitted only as input to impute_linear. Event IDs must
    be supplied by the caller to guarantee storm independence; None means unknown.
    """

    station_id: str
    timestamp: datetime
    water_level: float | None
    event_id: str | None = None


def _validate(records, *, allow_missing=False):
    records = list(records)
    previous = {}
    for record in records:
        if not isinstance(record, ObservationRecord):
            # Keep one exception contract for invalid observation inputs.
            raise ValueError(  # noqa: TRY004
                "Required fields: station_id, timestamp, water_level (ObservationRecord)"
            )
        if not isinstance(record.station_id, str) or not record.station_id.strip():
            raise ValueError("station_id must be a non-empty string")
        if record.event_id is not None and (
            not isinstance(record.event_id, str) or not record.event_id.strip()
        ):
            raise ValueError("event_id must be a non-empty string or None")
        time = record.timestamp
        if not isinstance(time, datetime) or time.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be a timezone-aware UTC datetime")
        if record.station_id in previous and time <= previous[record.station_id]:
            raise ValueError("Timestamps must be strictly monotonic per station")
        previous[record.station_id] = time
        value = record.water_level
        if value is None:
            if allow_missing:
                continue
            raise ValueError("water_level must be non-null")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("water_level must be numeric meters")  # noqa: TRY004
        if allow_missing and np.isnan(value):
            continue
        if not np.isfinite(value) or not -10 <= value <= 30:
            raise ValueError("water_level must be finite and within [-10, 30] meters")
    return records


def validate_records(records) -> None:
    """Validate records without sorting or mutating; reject duplicate station times."""
    _validate(records)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Raw partitions plus every excluded observation, for embargo accounting."""

    train: tuple[ObservationRecord, ...]
    val: tuple[ObservationRecord, ...]
    test: tuple[ObservationRecord, ...]
    excluded: tuple[ObservationRecord, ...]


def chronological_split(records, train_frac, val_frac, test_frac, embargo_days=7):
    """Split on distinct UTC timestamps shared across stations, before windows.

    Floor(n_times * train_frac) and floor(n_times * (train_frac + val_frac))
    select the first timestamps of validation and test. Drop [cut, cut+embargo)
    at each cut. Fractions describe pre-embargo time counts, not retained rows.
    Entire named events touching a gap or multiple partitions are excluded across
    all gauges. Reject empty retained partitions. Unknown events cannot be audited.
    Choose embargo to cover input history plus target horizon for later windows.
    """
    records = _validate(records)
    fractions = (train_frac, val_frac, test_frac)
    if any(not isinstance(f, Real) or not np.isfinite(f) or not 0 < f < 1
           for f in fractions) or not np.isclose(sum(fractions), 1, rtol=0, atol=1e-12):
        raise ValueError("Positive split fractions must sum to one")
    if not isinstance(embargo_days, Real) or not np.isfinite(embargo_days) or embargo_days < 0:
        raise ValueError("embargo_days must be finite and non-negative")
    times = sorted({r.timestamp for r in records})
    a, b = int(len(times) * train_frac), int(len(times) * (train_frac + val_frac))
    if not 0 < a < b < len(times):
        raise ValueError("Too few timestamps for three non-empty partitions")
    first, second = times[a], times[b]
    gap = timedelta(days=float(embargo_days))
    labels = []
    event_labels = defaultdict(set)
    for record in records:
        t = record.timestamp
        label = (0 if t < first else 1 if first + gap <= t < second
                 else 2 if t >= second + gap else 3)
        labels.append(label)
        if record.event_id is not None:
            event_labels[record.event_id].add(label)
    purged = {event for event, labels_ in event_labels.items()
              if 3 in labels_ or len(labels_) > 1}
    partitions = [[], [], [], []]
    for record, label in zip(records, labels):
        partitions[3 if record.event_id in purged else label].append(record)
    for partition in partitions:
        partition.sort(key=lambda r: (r.timestamp, r.station_id))
    # Timestamp fallback checks non-empty partitions without asserting that unknown
    # events are independent. Prefixes avoid collisions with caller-supplied IDs.
    assert_disjoint_groups(*(
        [f"event:{r.event_id}" if r.event_id is not None else f"time:{r.timestamp.isoformat()}"
         for r in partition] for partition in partitions[:3]
    ))
    return ChronologicalSplit(*(tuple(partition) for partition in partitions))


@dataclass(frozen=True)
class GaugeGap:
    station_id: str
    start: datetime
    end: datetime
    hours: float


def gauge_gaps(records, max_gap_hours=24) -> list[GaugeGap]:
    """Flag consecutive observed timestamp gaps strictly above the threshold."""
    records = _validate(records, allow_missing=True)
    if (not isinstance(max_gap_hours, Real) or not np.isfinite(max_gap_hours)
            or max_gap_hours < 0):
        raise ValueError("max_gap_hours must be finite and non-negative")
    previous = {}
    gaps = []
    for record in records:
        if record.station_id in previous:
            start = previous[record.station_id]
            hours = (record.timestamp - start).total_seconds() / 3600
            if hours > max_gap_hours:
                gaps.append(GaugeGap(record.station_id, start, record.timestamp, hours))
        previous[record.station_id] = record.timestamp
    return gaps


def impute_linear(records) -> list[ObservationRecord]:
    """Fill existing internal None/NaN rows using elapsed time within each gauge.

    Retrospective only: call separately on already split partitions. Never use
    this two-sided interpolation to construct operational forecast inputs.
    No new timestamps or extrapolation; reject unbracketed missing values and
    interpolation whose anchors have different event IDs from the missing row.
    """
    records = _validate(records, allow_missing=True)
    stations = defaultdict(list)
    for i, record in enumerate(records):
        stations[record.station_id].append(i)
    result = records.copy()
    for indices in stations.values():
        rows = [records[i] for i in indices]
        values = np.array([r.water_level for r in rows], dtype=float)
        good = np.flatnonzero(np.isfinite(values))
        missing = np.flatnonzero(~np.isfinite(values))
        if not len(missing):
            continue
        if len(good) < 2 or missing[0] < good[0] or missing[-1] > good[-1]:
            raise ValueError("Missing values must be bracketed; no extrapolation")
        times = np.array([(r.timestamp - rows[0].timestamp).total_seconds() for r in rows])
        for j in missing:
            position = np.searchsorted(good, j)
            left, right = good[position - 1], good[position]
            if not rows[left].event_id == rows[j].event_id == rows[right].event_id:
                raise ValueError("Cannot interpolate across event boundaries")
            value = np.interp(times[j], times[[left, right]], values[[left, right]])
            result[indices[j]] = replace(rows[j], water_level=float(value))
    validate_records(result)
    return result


def make_synthetic_records(n_stations=3, days=30, freq_min=60):
    """Deterministic, fictional hourly-style meter observations (local RNG seed 0).

    Interval is [2020-01-01 UTC, start+days); daily IDs are fixture groups,
    not scientifically defined independent storms. No external source data.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
           for v in (n_stations, days, freq_min)):
        raise ValueError("n_stations, days and freq_min must be positive integers")
    rng = np.random.default_rng(0)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    records = []
    for minute in range(0, days * 24 * 60, freq_min):
        for station in range(n_stations):
            level = 1.5 * np.sin(2 * np.pi * minute / (12 * 60)) + rng.normal(0, 0.1)
            records.append(ObservationRecord(
                f"synthetic-{station}", start + timedelta(minutes=minute),
                float(np.clip(level, -10, 30)), f"synthetic-day-{minute // 1440}",
            ))
    return records
