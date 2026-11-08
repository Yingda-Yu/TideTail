"""QC pipeline regression tests on synthetic gauge data."""
import numpy as np

from tidetail.data import GaugeRecord
from tidetail.qc import qc_pipeline
from tidetail.synthetic import make_toy_gauge


def test_qc_detects_duplicates():
    g = make_toy_gauge("S1", n_days=3, seed=7)
    # inject a duplicate
    dup_time = g.time.copy()
    dup_time[50] = dup_time[51]  # make two rows identical
    dup_wl = g.water_level.copy()
    g2 = GaugeRecord(station_id=g.station_id, time=dup_time, water_level=dup_wl,
                     sampling_period=g.sampling_period)
    cleaned, summary = qc_pipeline(g2)
    assert summary.n_duplicates >= 1
    # cleaned should have fewer or equal entries
    assert cleaned.n <= g2.n


def test_qc_reports_missing_fraction():
    g = make_toy_gauge("S1", n_days=5, seed=42, include_gaps=True,
                       include_nans=False, include_outliers=False)
    _, summary = qc_pipeline(g)
    assert summary.missing_fraction > 0.0  # we injected gaps


def test_qc_marks_outliers():
    g = make_toy_gauge("S1", n_days=10, seed=11, include_outliers=True,
                       include_gaps=False, include_nans=False)
    _, summary = qc_pipeline(g)
    # outlier_iqr or outlier_3sigma should catch at least one injected spike
    assert (summary.n_outliers_3sigma + summary.n_outliers_iqr) >= 1


def test_qc_bounds_clamps_physical_oob():
    g = make_toy_gauge("S1", n_days=3, seed=9, include_gaps=False,
                       include_outliers=False, include_nans=False)
    # Inject a clearly impossible value
    g.water_level[10] = 100.0
    _, summary = qc_pipeline(g, bounds=(-5.0, 15.0))
    assert summary.n_bounds_clamped == 1


def test_qc_interpolates_short_gaps():
    """Interpolator fills short gaps and leaves long ones untouched."""
    g = make_toy_gauge("S1", n_days=5, seed=3, include_gaps=False,
                       include_nans=False, include_outliers=False)
    # Inject a contiguous NaN run of length 3 (1800s) and 15 (9000s)
    g.water_level = g.water_level.copy()
    g.water_level[100:103] = np.nan   # gap of 1800s
    g.water_level[300:315] = np.nan   # gap of 9000s
    cleaned, _ = qc_pipeline(g, max_interp_gap_s=1800.0)
    # Short gap filled → finite around 100-103
    assert np.isfinite(cleaned.water_level[100:103]).all()
    # Long gap (> 1800s) stays NaN
    assert not np.isfinite(cleaned.water_level[300:315]).all()
