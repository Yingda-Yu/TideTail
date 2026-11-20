"""Hand-computable unit tests for evaluate.py metrics.

Every test uses small arrays where the expected answer can be obtained with
mental arithmetic. This catches formula bugs (wrong denominator, swapped TP/FP,
etc.) that black-box numerical tests would miss.

All tests run with PYTHONPATH=src python3 -m pytest tests/test_evaluate.py -v
"""
from __future__ import annotations

import numpy as np

from tidetail.evaluate import (
    bootstrap_ci,
    lead_time_metrics,
    mae,
    paired_bootstrap_diff,
    peak_magnitude_error,
    peak_timing_error,
    rmse,
    tail_mae,
    tail_rmse,
    threshold_metrics,
)

# ==========================================================================
# Cell-level scalar metrics
# ==========================================================================


class TestScalarMetrics:
    def test_mae_perfect(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mae_hand(self):
        # |(1-3) + (2-2) + (3-1)| / 3 = (2+0+2)/3 = 4/3
        assert abs(mae([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) - 4 / 3) < 1e-12

    def test_mae_all_nan(self):
        assert np.isnan(mae([np.nan, np.nan], [1, 2]))

    def test_mae_partial_nan(self):
        # y[0]=NaN excluded (not in denominator); y[1]=2, yhat[1]=2 → |2-2| = 0
        assert abs(mae([np.nan, 2.0], [1.0, 2.0]) - 0.0) < 1e-12

    def test_rmse_hand(self):
        # sqrt(((1-3)^2 + (2-2)^2 + (3-1)^2)/3) = sqrt((4+0+4)/3) = sqrt(8/3)
        expected = np.sqrt(8 / 3)
        assert abs(rmse([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) - expected) < 1e-12

    def test_tail_mae_tau_q95(self):
        # y=[1,2,3,4,5], yhat=[0,1,2,3,4], tau=4.5 (tail = y>=4.5 → cell 5)
        # |5-4| = 1
        assert abs(tail_mae([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], train_threshold=4.5) - 1.0) < 1e-12

    def test_tail_rmse_hand(self):
        # same tail as above
        expected = 1.0  # sqrt(|5-4|^2/1)
        assert abs(tail_rmse([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], train_threshold=4.5) - expected) < 1e-12

    def test_tail_empty_returns_nan(self):
        assert np.isnan(tail_mae([1, 2, 3], [0, 0, 0], train_threshold=10.0))


# ==========================================================================
# Threshold metrics — contingency table hand-verifiable
# ==========================================================================


class TestThresholdMetrics:
    """
    Contingency table for tau = 3 on finite cells only:

    y = [0, 1, 4, 5, 2, 4, np.nan]
    ŷ = [0, 0, 4, 2, 4, 5, 1]

    Only first 6 cells (all finite):
      i=0: y=0<3, ŷ=0<3 → TN
      i=1: y=1<3, ŷ=0<3 → TN
      i=2: y=4≥3, ŷ=4≥3 → TP
      i=3: y=5≥3, ŷ=2<3 → FN
      i=4: y=2<3, ŷ=4≥3 → FP
      i=5: y=4≥3, ŷ=5≥3 → TP

    TP=2, FP=1, FN=1, TN=2, N_finite=6
    Precision = 2/(2+1) = 2/3
    Recall    = 2/(2+1) = 2/3
    F1        = 2*0.667*0.667/(0.667+0.667) = 2/3
    FPR       = 1/(1+2) = 1/3
    FAR       = 1/(2+1) = 1/3
    """

    def test_full_contingency(self):
        y = np.array([0.0, 1.0, 4.0, 5.0, 2.0, 4.0, np.nan])
        yhat = np.array([0.0, 0.0, 4.0, 2.0, 4.0, 5.0, 1.0])
        tau = 3.0
        m = threshold_metrics(y, yhat, threshold=tau)
        # Finite cells only (indices 0-5):
        #   i=0: y=0<3, ŷ=0<3 → TN
        #   i=1: y=1<3, ŷ=0<3 → TN
        #   i=2: y=4≥3, ŷ=4≥3 → TP
        #   i=3: y=5≥3, ŷ=2<3 → FN
        #   i=4: y=2<3, ŷ=4≥3 → FP
        #   i=5: y=4≥3, ŷ=5≥3 → TP
        assert m["tp"] == 2.0  # i=2, i=5
        assert m["fp"] == 1.0  # i=4
        assert m["fn"] == 1.0  # i=3
        assert m["tn"] == 2.0  # i=0, i=1 (NaN at i=6 excluded)
        assert m["n_finite"] == 6.0
        assert abs(m["precision"] - 2 / 3) < 1e-12
        assert abs(m["recall"] - 2 / 3) < 1e-12
        assert abs(m["f1"] - 2 / 3) < 1e-12
        assert abs(m["fpr"] - 1 / 3) < 1e-12  # FP/(FP+TN) = 1/3
        assert abs(m["far"] - 1 / 3) < 1e-12  # FP/(TP+FP) = 1/3
        assert m["threshold"] == 3.0

    def test_no_finite_cells(self):
        m = threshold_metrics([np.nan], [np.nan], threshold=1.0)
        for k in ["precision", "recall", "f1", "fpr", "far", "tp", "fp", "fn", "tn"]:
            assert np.isnan(m[k]), f"{k} should be NaN with no finite cells"

    def test_no_events_but_all_cells_finite(self):
        # tau=10, no cells exceed it. Predicted all 0.
        m = threshold_metrics([1, 2, 3], [0, 0, 0], threshold=10.0)
        assert m["tp"] == 0.0 and m["fp"] == 0.0 and m["fn"] == 0.0 and m["tn"] == 3.0
        # Precision: NaN (no alarms)
        assert np.isnan(m["precision"])
        # Recall: NaN (no real events)
        assert np.isnan(m["recall"])
        # FPR: 0/3 = 0 (no false alarms, no events)
        assert abs(m["fpr"] - 0.0) < 1e-12
        # FAR: NaN (no alarms)
        assert np.isnan(m["far"])

    def test_all_alarms_no_events(self):
        # tau=2, y=[0,0,0] all < 2; yhat=[5,5,5] all >= 2
        m = threshold_metrics([0, 0, 0], [5, 5, 5], threshold=2.0)
        assert m["tp"] == 0.0 and m["fp"] == 3.0 and m["fn"] == 0.0 and m["tn"] == 0.0
        # Precision = TP/(TP+FP) = 0/(0+3) = 0.0 — correct, not NaN
        assert m["precision"] == 0.0
        # Recall = TP/(TP+FN) = 0/(0+0) → NaN (division by zero)
        assert np.isnan(m["recall"])


# ==========================================================================
# Event-level metrics
# ==========================================================================


class TestEventMetrics:
    def test_peak_magnitude_hand(self):
        # max(y)=5, max(ŷ)=3, |5-3|=2
        assert peak_magnitude_error([1, 5, 2, 4], [0, 3, 0, 2]) == 2.0

    def test_peak_timing_hand(self):
        # argmax(y)=1 (max at index 1 = 5), argmax(ŷ)=3 (max at index 3 = 4), |1-3|=2
        assert peak_timing_error([1, 5, 2, 3], [0, 2, 1, 4]) == 2.0

    def test_peak_timing_all_nan(self):
        assert np.isnan(peak_timing_error([np.nan], [np.nan]))

    def test_peak_timing_tie_takes_first(self):
        # both have ties at indices 0 and 1, argmax returns first → both 0, error 0
        assert peak_timing_error([3, 3, 1], [3, 2, 0]) == 0.0


# ==========================================================================
# Lead-time and station breakdowns
# ==========================================================================


class TestStructuralBreakdowns:
    def test_lead_time_perfect(self):
        y = np.array([[1, 2, 3], [4, 5, 6]])  # 2 forecasts × 3 horizons
        r = lead_time_metrics(y, y)
        assert r["horizon"] == 3
        np.testing.assert_array_equal(r["mae"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(r["rmse"], [0.0, 0.0, 0.0])

    def test_lead_time_errors(self):
        y = np.array([[1, 2, 3], [4, 5, 6]])
        yhat = np.array([[1, 3, 3], [4, 5, 8]])  # errors: h1=0,h2=1,h3=0 and h1=0,h2=0,h3=2
        r = lead_time_metrics(y, yhat)
        # mae[0] = (0+0)/2=0, mae[1] = (1+0)/2=0.5, mae[2] = (0+2)/2=1
        assert abs(r["mae"][0] - 0.0) < 1e-12
        assert abs(r["mae"][1] - 0.5) < 1e-12
        assert abs(r["mae"][2] - 1.0) < 1e-12
        # rmse[2] = sqrt((0^2 + 2^2)/2) = sqrt(2)
        assert abs(r["rmse"][2] - np.sqrt(2)) < 1e-12


# ==========================================================================
# Bootstrap CI
# ==========================================================================


class TestBootstrapCI:
    def test_too_few_events_returns_nan(self):
        ci = bootstrap_ci([1.0, 2.0], n_boot=50, seed=42)
        assert np.isnan(ci["mean"]) and np.isnan(ci["lo"]) and np.isnan(ci["hi"])
        assert ci["n_events"] == 2.0

    def test_consistent_with_fixed_seed(self):
        # Deterministic seed should produce reproducible results
        ci1 = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200, seed=42)
        ci2 = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200, seed=42)
        assert ci1["mean"] == ci2["mean"]
        assert ci1["lo"] == ci2["lo"]
        assert ci1["hi"] == ci2["hi"]

    def test_paired_bootstrap_hand(self):
        # Two methods on 4 storms: A=[1,2,3,4], B=[2,3,4,5]. diffs = [-1,-1,-1,-1]
        # mean diff = -1.0
        result = paired_bootstrap_diff([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0],
                                       n_boot=500, seed=7)
        assert abs(result["diff_mean"] - (-1.0)) < 1e-12
        assert result["n_pairs"] == 4.0
        # All resampled means of constant -1 are exactly -1
        assert abs(result["diff_lo"] - (-1.0)) < 1e-12
        assert abs(result["diff_hi"] - (-1.0)) < 1e-12


# ==========================================================================
# Reproduce the threshold_metrics hand-calc from audit report
# ==========================================================================


def test_threshold_metrics_audit_example():
    """Original audit probe from PR #19: FP/(FP+TN) as FPR — must be 1/3."""
    y = np.array([0.0, 1.0, 4.0, 5.0, 2.0, 4.0, np.nan])
    yhat = np.array([0.0, 0.0, 4.0, 2.0, 4.0, 5.0, 1.0])
    m = threshold_metrics(y, yhat, threshold=3.0)
    # FPR = FP/(FP+TN) = 1/(1+2) = 0.333 — must NOT be FP/N = 1/6
    assert abs(m["fpr"] - 1 / 3) < 1e-12, "FPR must use FP/(FP+TN), not FP/N"
    # FAR = FP/(TP+FP) = 1/(2+1) = 0.333
    assert abs(m["far"] - 1 / 3) < 1e-12, "FAR must use FP/(TP+FP)"
