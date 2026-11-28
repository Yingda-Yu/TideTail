"""Wave 3.2.1 regression tests — Bonferroni survivor rule (no double
correction) and MC standard-error floor.

The Wave 3.2 architecture-interaction correction compared the ADJUSTED
Bonferroni p against alpha/n (a double correction). The correct rule is
exactly one of the two equivalent forms:

    raw p < alpha / n          (equivalently)
    adjusted p = min(1, n * raw) < alpha

These tests pin the rule for future edits.
"""
import math

ALPHA = 0.05
N = 10


def _survivor_rule(raw_p: float) -> bool:
    """Reference (correct) Bonferroni rule."""
    return raw_p < ALPHA / N


def _wrong_double_corrected_rule(raw_p: float) -> bool:
    """The erroneous Wave 3.2 rule: adjusted p compared against alpha/n."""
    adj = min(1.0, N * raw_p)
    return adj < ALPHA / N


def test_rules_are_equivalent_on_grid():
    # raw < alpha/n  <=>  min(1, n*raw) < alpha  over a dense grid
    for i in range(1, 2000):
        raw = i / 50000
        adj = min(1.0, N * raw)
        assert _survivor_rule(raw) == (adj < ALPHA)


def test_double_correction_rejects_too_many():
    # SE-Australia tail-aware: raw p ~ 0.0008, adjusted ~ 0.008
    raw = 0.0008
    adj = min(1.0, N * raw)
    assert math.isclose(adj, 0.008, rel_tol=1e-9)
    assert _survivor_rule(raw) is True            # correct: survives
    assert _wrong_double_corrected_rule(raw) is False  # bug: dropped it


def test_double_correction_is_strictly_stricter():
    # any p rejected by the correct rule is also rejected by the buggy rule;
    # the buggy rule additionally rejects everything in [alpha/n^2, alpha/n)
    for i in range(1, 600):
        raw = i / 100000
        if _wrong_double_corrected_rule(raw):
            assert _survivor_rule(raw)
    raw_mid = ALPHA / N / 2  # 0.0025: correct -> survive; buggy -> reject
    assert _survivor_rule(raw_mid) is True
    assert _wrong_double_corrected_rule(raw_mid) is False


def test_mc_standard_error_floor_at_reported_resolution():
    # p ~ 0.0001 from n_boot=10000 with (b+1)/(n_boot+1) floor:
    # binomial SE = sqrt(p(1-p)/n) ~ 1e-4, NOT 1e-5.
    p, n = 1e-4, 10000
    se = math.sqrt(p * (1 - p) / n)
    assert 0.5e-4 < se < 2e-4
    assert math.isclose(se, 1e-4, rel_tol=0.35)
