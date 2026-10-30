"""Synthetic-data deterministic behaviour."""
import numpy as np

from tidetail.synthetic import make_toy_gauge, make_toy_tide_series


def test_toy_series_deterministic():
    t1, c1, l1 = make_toy_tide_series(n_days=7, seed=0)
    t2, c2, l2 = make_toy_tide_series(n_days=7, seed=0)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_allclose(c1, c2)
    np.testing.assert_allclose(l1, l2)


def test_toy_series_has_extreme_events():
    t, clean, _ = make_toy_tide_series(n_days=60, seed=0, include_storm=True)
    # storm bumps push surge far above M2 amplitude
    assert np.nanmax(clean) > 3.0  # 1.2 M2 + K1 + storm bump


def test_toy_gauge_has_missing_and_outliers():
    g = make_toy_gauge("S1", n_days=14, seed=1)
    assert np.any(~np.isfinite(g.water_level))  # NaNs present
    assert g.n > 100
