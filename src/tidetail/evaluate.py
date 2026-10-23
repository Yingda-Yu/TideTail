"""Shared forecasting evaluation — real-only, no synthetic in val/test."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m])))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def tail_error(y: np.ndarray, yhat: np.ndarray,
               train_threshold: float, tail_side: str = "high") -> float:
    """MAE restricted to events where |y| exceeds train-only threshold."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if tail_side == "high":
        m = m & (y >= train_threshold)
    else:
        m = m & (y <= -train_threshold)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m])))


def peak_magnitude_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """|peak_y - peak_yhat|."""
    return float(abs(np.nanmax(y) - np.nanmax(yhat)))


def peak_timing_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """|argmax(y) - argmax(yhat)| in steps."""
    if y.size == 0 or yhat.size == 0:
        return float("nan")
    return float(abs(int(np.nanargmax(y)) - int(np.nanargmax(yhat))))


def threshold_metrics(y: np.ndarray, yhat: np.ndarray,
                      threshold: float) -> Dict[str, float]:
    """Precision/recall/false_alarm at a threshold (train-only by protocol)."""
    pred_event = yhat >= threshold
    true_event = y >= threshold
    tp = float(((pred_event) & (true_event)).sum())
    fp = float(((pred_event) & (~true_event)).sum())
    fn = float(((~pred_event) & (true_event)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    false_alarm_rate = fp / float(y.size) if y.size > 0 else float("nan")
    return {
        "precision": precision, "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "tp": tp, "fp": fp, "fn": fn,
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 200,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """95% bootstrap CI for the mean of `values`."""
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
             for _ in range(n_boot)]
    boots.sort()
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return {"mean": float(np.mean(arr)), "lo": float(lo), "hi": float(hi)}
