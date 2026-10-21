from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import pairwise

import numpy as np
import pytest

from tidetail.data import (
    ObservationRecord,
    chronological_split,
    gauge_gaps,
    impute_linear,
    make_synthetic_records,
    validate_records,
)

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def row(hour=0, value=1.0, station="a", event=None):
    return ObservationRecord(station, START + timedelta(hours=hour), value, event)


def test_valid_records_and_physical_endpoints():
    validate_records([row(value=-10), row(1, 30), row(station="b")])
    validate_records([])


@pytest.mark.parametrize("record", [
    row(value=None), row(value=np.nan), row(value=np.inf), row(value=-10.01),
    row(value=30.01), row(value="1"), row(value=True), row(station=""),
    replace(row(), timestamp=START.replace(tzinfo=None)),
    replace(row(), timestamp=START.astimezone(timezone(timedelta(hours=1)))),
    replace(row(), timestamp="2020-01-01"), replace(row(), event_id=""),
    {"station_id": "a", "timestamp": START},
])
def test_invalid_record(record):
    with pytest.raises(ValueError):
        validate_records([record])


@pytest.mark.parametrize("records", [[row(1), row()], [row(), row()]])
def test_nonmonotonic_or_duplicate_station_time(records):
    with pytest.raises(ValueError, match="monotonic"):
        validate_records(records)
    with pytest.raises(ValueError, match="monotonic"):
        chronological_split(records, .6, .2, .2)


def test_split_shared_times_embargo_and_accounting():
    records = make_synthetic_records(days=60)
    split = chronological_split(records, .6, .2, .2)
    parts = (split.train, split.val, split.test)
    for left, right in pairwise(parts):
        assert min(r.timestamp for r in right) - max(r.timestamp for r in left) >= timedelta(days=7)
    for part in parts:
        validate_records(part)
        assert list(part) == sorted(part, key=lambda r: (r.timestamp, r.station_id))
    assert set().union(*(set(part) for part in (*parts, split.excluded))) == set(records)
    assert sum(map(len, (*parts, split.excluded))) == len(records)
    time_sets = [{r.timestamp for r in part} for part in parts]
    event_sets = [{r.event_id for r in part} for part in parts]
    for sets in (time_sets, event_sets):
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_fraction_cuts_use_unique_times_not_station_density():
    records = [row(i) for i in range(10)]
    # Input can be grouped by station; only each station's ordering is required.
    dense = records + [row(i, station="b") for i in range(5)]
    split = chronological_split(dense, .6, .2, .2, embargo_days=0)
    assert min(r.timestamp for r in split.val) == START + timedelta(hours=6)
    assert min(r.timestamp for r in split.test) == START + timedelta(hours=8)
    assert len(split.excluded) == 0


def test_embargo_exact_boundary_included():
    records = [row(i) for i in range(20)]
    split = chronological_split(records, .5, .25, .25, embargo_days=1 / 24)
    assert [r.timestamp for r in split.excluded] == [row(10).timestamp, row(15).timestamp]
    assert split.val[0].timestamp == row(11).timestamp
    assert split.test[0].timestamp == row(16).timestamp


def test_event_crossing_cut_purged_whole_across_stations():
    records = [row(i, event="crossing" if i in (5, 6) else f"day-{i}")
               for i in range(10)]
    records += [row(4, station="b", event="crossing")]
    split = chronological_split(records, .6, .2, .2, embargo_days=0)
    assert {r.event_id for r in split.excluded} == {"crossing"}
    assert len(split.excluded) == 3
    assert all(r.event_id != "crossing" for p in (split.train, split.val, split.test) for r in p)


def test_event_touching_embargo_purged_even_on_train_side():
    records = [row(i, event="crossing" if i in (9, 10) else f"hour-{i}")
               for i in range(20)]
    split = chronological_split(records, .5, .25, .25, embargo_days=1 / 24)
    assert row(9, event="crossing") in split.excluded
    assert row(10, event="crossing") in split.excluded


@pytest.mark.parametrize("fractions", [(.5, .2, .2), (0, .5, .5), (-.1, .5, .6),
                                        (.5, .5, np.nan), (.5, .5, np.inf)])
def test_invalid_split_fractions(fractions):
    with pytest.raises(ValueError, match="fractions"):
        chronological_split(make_synthetic_records(), *fractions)


@pytest.mark.parametrize("embargo", [-1, np.inf, np.nan])
def test_invalid_embargo(embargo):
    with pytest.raises(ValueError, match="embargo"):
        chronological_split(make_synthetic_records(), .6, .2, .2, embargo)


def test_too_little_data_or_excessive_embargo_fails():
    for records, embargo in [([], 0), ([row()], 0), (make_synthetic_records(), 100)]:
        with pytest.raises(ValueError):
            chronological_split(records, .6, .2, .2, embargo)


def test_gaps_per_station_and_strict_threshold():
    records = [row(), row(24), row(49), row(station="b"), row(30, station="b")]
    gaps = gauge_gaps(records)
    assert [(g.station_id, g.hours) for g in gaps] == [("a", 25), ("b", 30)]
    assert gaps[0].start == row(24).timestamp
    assert gaps[0].end == row(49).timestamp
    assert gauge_gaps([]) == []


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf])
def test_invalid_gap_threshold(threshold):
    with pytest.raises(ValueError):
        gauge_gaps([], threshold)


def test_imputation_uses_elapsed_time_per_station_without_mutation():
    records = [row(0, 0), row(1, None), row(4, 8), row(0, 10, "b"),
               row(2, np.nan, "b"), row(4, 6, "b")]
    filled = impute_linear(records)
    assert [r.water_level for r in filled] == [0, 2, 8, 10, 8, 6]
    assert records[1].water_level is None
    assert np.isnan(records[4].water_level)
    assert [(r.station_id, r.timestamp, r.event_id) for r in filled] == [
        (r.station_id, r.timestamp, r.event_id) for r in records]
    validate_records(filled)


@pytest.mark.parametrize("records", [[row(value=None)], [row(value=None), row(1)],
                                       [row(), row(1, None)], [row(value=np.inf)]])
def test_imputation_rejects_unbracketed_or_invalid_values(records):
    with pytest.raises(ValueError):
        impute_linear(records)


def test_imputation_rejects_event_crossing():
    with pytest.raises(ValueError, match="event boundaries"):
        impute_linear([row(0, 0, event="a"), row(1, None, event="a"), row(2, 2, event="b")])


def test_imputation_no_insertions_and_empty():
    records = [row(), row(48)]
    assert impute_linear(records) == records
    assert impute_linear([]) == []


def test_synthetic_fixture_determinism_and_global_rng_independence():
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    records = make_synthetic_records()
    assert np.random.random() == expected
    assert records == make_synthetic_records()
    assert len(records) == 3 * 30 * 24
    assert len({r.station_id for r in records}) == 3
    validate_records(records)
    assert len(make_synthetic_records(1, 1, 100)) == 15


@pytest.mark.parametrize("args", [(0, 1, 60), (1, -1, 60), (1, 1, 0), (1, 1, 1.5)])
def test_invalid_fixture_configuration(args):
    with pytest.raises(ValueError):
        make_synthetic_records(*args)
