"""Issue #18: adversarial checks missing from the original split tests."""
import pytest

from tidetail.splits import EventGroup, partition_chronological


def test_reject_overlapping_event_intervals():
    groups = [EventGroup('a', 0, 200), EventGroup('b', 100, 150),
              EventGroup('c', 300, 350)]
    with pytest.raises(ValueError, match='overlap'):
        partition_chronological(groups, .34, .33)


def test_reject_duplicate_event_ids():
    groups = [EventGroup('same', 0, 10), EventGroup('same', 20, 30),
              EventGroup('c', 40, 50)]
    with pytest.raises(ValueError, match='Duplicate'):
        partition_chronological(groups, .34, .33)


@pytest.mark.parametrize('start,end', [(-1, 10), (20, 10)])
def test_reject_invalid_intervals(start, end):
    groups = [EventGroup('a', start, end), EventGroup('b', 30, 40),
              EventGroup('c', 50, 60)]
    with pytest.raises(ValueError, match='interval'):
        partition_chronological(groups, .34, .33)


def test_accept_adjacent_disjoint_intervals():
    groups = [EventGroup('a', 0, 10), EventGroup('b', 11, 20),
              EventGroup('c', 21, 30)]
    train, val, test = partition_chronological(groups, .34, .33)
    assert [g.event_id for fold in (train, val, test) for g in fold] == ['a', 'b', 'c']
