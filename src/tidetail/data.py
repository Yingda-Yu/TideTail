"""Tide/gauge observation data handling.

Loads time-indexed gauge observations, aligns units and frequencies,
detects gaps and outliers, and separates tide vs. surge components.
Everything works on NumPy arrays so we don't depend on pandas
until the user installs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np


@dataclass
class GaugeRecord:
    """A single gauge station's time-indexed water-level series."""

    station_id: str
    units: str = "m"          # metres relative to gauge datum
    sampling_period: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    lat: float | None = None
    lon: float | None = None
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[s]"))
    water_level: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.water_level.size)

    @property
    def start(self) -> np.datetime64 | None:
        return self.time[0] if self.n else None

    @property
    def end(self) -> np.datetime64 | None:
        return self.time[-1] if self.n else None


def time_intervals(time: np.ndarray) -> np.ndarray:
    """Return consecutive time-step durations as timedelta64[s]."""
    if time.size < 2:
        return np.array([], dtype="timedelta64[s]")
    return np.diff(time.astype("datetime64[s]"))


def gap_mask(time: np.ndarray, expected: timedelta) -> np.ndarray:
    """Boolean mask — True where a subsequent sample is missing."""
    if time.size < 2:
        return np.array([], dtype=bool)
    expected_s = int(expected.total_seconds())
    dt = time_intervals(time).astype("timedelta64[s]").astype(float)
    return dt > expected_s * 1.5  # > 1.5x expected = missing step(s)


def missing_ratio(time: np.ndarray, expected: timedelta) -> float:
    """Fraction of expected samples that are absent (gap+edge)."""
    if time.size < 2:
        return 1.0
    span_s = (time[-1] - time[0]).astype("timedelta64[s]").astype(float)
    expected_n = int(span_s / expected.total_seconds()) + 1
    return max(0.0, 1.0 - float(time.size) / float(expected_n))


def outlier_3sigma(values: np.ndarray) -> np.ndarray:
    """Boolean mask — True where |z| > 3 on the full series."""
    if values.size < 20:
        return np.zeros(values.shape, dtype=bool)
    mu = np.nanmean(values)
    sd = np.nanstd(values)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros(values.shape, dtype=bool)
    return np.abs(values - mu) > 3 * sd


def outlier_iqr(values: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Boolean mask — True where outside [Q1 - kIQR, Q3 + kIQR]."""
    if values.size < 20:
        return np.zeros(values.shape, dtype=bool)
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(values.shape, dtype=bool)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (values < lo) | (values > hi)


def linear_interpolate(values: np.ndarray, max_gap_s: float = 3600,
                       time_s: np.ndarray | None = None) -> np.ndarray:
    """Linear interpolate NaNs / outliers, only gaps <= max_gap_s.

    If time_s is None, assumes uniform gaps and interpolates up to
    max_gap_s / mean_dt consecutive NaNs.  Longer gaps are left NaN.
    """
    out = values.copy()
    mask = ~np.isfinite(out)
    if not mask.any():
        return out
    if time_s is None:
        # count consecutive NaN runs
        run_len = np.zeros(out.size, dtype=int)
        for i in range(1, out.size):
            if mask[i]:
                run_len[i] = run_len[i - 1] + 1
        dt_est = max_gap_s  # assume worst case; user should supply time_s
        max_steps = max(1, int(max_gap_s / dt_est))
        mask = mask & (run_len <= max_steps)
        out[mask] = np.interp(np.where(mask)[0],
                              np.where(~mask)[0], out[~mask],
                              left=np.nan, right=np.nan)
        return out
    # time-aware version
    idx = np.where(mask)[0]
    for i in idx:
        # find gap length
        j = i
        while j + 1 < out.size and not np.isfinite(out[j + 1]):
            j += 1
        gap_s = float(time_s[j] - time_s[i - 1]) if i > 0 else 0.0
        if i == 0 or j == out.size - 1 or gap_s > max_gap_s:
            continue
        out[i:j + 1] = np.linspace(out[i - 1], out[j + 1], j - i + 1)
    return out


def detide_harmonic(values: np.ndarray, time_s: np.ndarray,
                    constituents: tuple[float, ...] = (2 * np.pi / 43200.0,  # M2
                                                       2 * np.pi / 86400.0,  # K1
                                                       2 * np.pi / 18050.0,  # S2
                                                       )) -> tuple[np.ndarray, np.ndarray]:
    """Linear regression fit of tidal harmonics → (tide_fit, surge_residual).

    constituents: angular frequencies (rad/s) of major tidal components.
    Returns (tide_fit, surge) where surge = values - tide_fit.
    """
    t = time_s - float(time_s[0])
    cols = [np.ones_like(t)]
    for w in constituents:
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    X = np.column_stack(cols)
    # Fit ignoring NaNs
    finite = np.isfinite(values)
    if finite.sum() < X.shape[1] + 2:
        return np.zeros_like(values), values.copy()
    beta, *_ = np.linalg.lstsq(X[finite], values[finite], rcond=None)
    tide_fit = X @ beta
    surge = values - tide_fit
    return tide_fit, surge
