"""NumPy evaluation primitives with explicit finite-data and empty-event rules."""

import numpy as np


def _pair(y_true, y_pred):
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1 or not truth.size:
        raise ValueError("Expected matching, non-empty 1D arrays")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("All metric inputs must be finite")
    return truth, prediction


def rmse(y_true, y_pred) -> float:
    """Root mean squared error over all paired observations (meters)."""
    truth, prediction = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error over all paired observations (meters)."""
    truth, prediction = _pair(y_true, y_pred)
    return float(np.mean(np.abs(prediction - truth)))


def tail_rmse(y_true, y_pred, q=0.95, *, train_values=None, threshold=None) -> float:
    """RMSE where truth >= a fixed training threshold; NaN if no exceedances.

    Supply exactly one of train_values (quantile fitted here) or a previously
    frozen threshold. Never estimate the cutoff on validation/test targets.
    q uses NumPy's linear quantile convention. Denominator is selected time points,
    not independent events; this function does not estimate uncertainty.
    """
    truth, prediction = _pair(y_true, y_pred)
    if not np.isscalar(q) or not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be between zero and one")
    if (train_values is None) == (threshold is None):
        raise ValueError("Supply exactly one of train_values or a frozen training threshold")
    if train_values is not None:
        training, _ = _pair(train_values, train_values)
        threshold = float(np.quantile(training, q))
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    mask = truth >= threshold
    return rmse(truth[mask], prediction[mask]) if np.any(mask) else float("nan")


def event_peak_error(y_true, y_pred, threshold) -> float:
    """Mean absolute peak magnitude error in true contiguous exceedance runs.

    threshold must be frozen from train. Each true run (truth >= threshold) is
    weighted equally; compare true/predicted maxima over that SAME interval.
    NaN means no observed events. Call on one ordered station/continuous segment
    at a time. This is neither peak timing error nor a false-alarm metric.
    """
    truth, prediction = _pair(y_true, y_pred)
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    edges = np.diff(np.r_[False, truth >= threshold, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    errors = [abs(np.max(truth[a:b]) - np.max(prediction[a:b]))
              for a, b in zip(starts, stops)]
    return float(np.mean(errors)) if errors else float("nan")
