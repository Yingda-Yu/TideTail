"""Research-integrity validation primitives."""
from collections.abc import Iterable


def assert_disjoint_groups(train: Iterable[str], val: Iterable[str], test: Iterable[str]) -> None:
    """Reject independent events shared across partitions."""
    a, b, c = set(train), set(val), set(test)
    if a & b or a & c or b & c:
        raise ValueError("Event-group leakage between train, validation and test")
    if not (a and b and c):
        raise ValueError("Every partition must contain at least one event")
