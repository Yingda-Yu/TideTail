import pytest

from tidetail.protocol import assert_disjoint_groups


def test_disjoint_events():
    assert_disjoint_groups(["a"], ["b"], ["c"])


@pytest.mark.parametrize("groups", [(["a"], ["a"], ["b"]), (["a"], ["b"], ["a"]), (["a"], ["b"], ["b"])])
def test_reject_shared_events(groups):
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_groups(*groups)


def test_reject_empty_partition():
    with pytest.raises(ValueError, match="at least one"):
        assert_disjoint_groups(["a"], [], ["c"])
