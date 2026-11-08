"""Synthetic toy data for unit tests and CPU smoke tests.

Generates:
  - tide + surge water level series with known extreme events
  - multi-gauge lagged propagation
  - controlled NaN/outlier/gap contamination

Everything deterministic when seeded.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from .data import GaugeRecord


def make_toy_tide_series(n_days: int = 30, period_minutes: int = 10,
                         seed: int = 0, include_storm: bool = True,
                         include_gaps: bool = True, include_outliers: bool = True,
                         include_nans: bool = True,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, clean_water_level, contaminated_water_level).

    Clean = deterministic tide + AR(2) surge noise + optional storm bumps.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24 * 60 // period_minutes
    dt_s = period_minutes * 60
    t = np.arange(n, dtype=float) * dt_s
    start = np.datetime64("2025-01-01T00:00:00")
    time = start + t.astype("timedelta64[s]")

    # Tidal components (metres)
    tide = (1.2 * np.sin(2 * np.pi * t / 43200.0)      # M2 semidiurnal
            + 0.5 * np.sin(2 * np.pi * t / 86400.0)   # K1 diurnal
            + 0.3 * np.sin(2 * np.pi * t / 18050.0)) # S2

    # AR(2) surge noise
    surge = np.zeros(n)
    for i in range(2, n):
        surge[i] = 0.6 * surge[i - 1] - 0.1 * surge[i - 2] + rng.normal(0, 0.08)

    level_clean = tide + surge

    # Storm bumps (localised positive residual)
    if include_storm:
        storms = [(0.45, 43200, 1.8), (0.72, 72000, 2.2), (0.92, 28800, 2.6)]
        for frac, dur_s, amp in storms:
            centre = int(frac * n)
            span = int(dur_s / dt_s)
            sigma = span / 3
            idx = np.arange(max(0, int(centre - 2*sigma)), min(n, int(centre + 2*sigma)), dtype=int)
            bump = amp * np.exp(-0.5 * ((idx - centre) / sigma) ** 2)
            level_clean[idx] += bump

    level = level_clean.copy()

    # NaNs
    if include_nans:
        nan_idx = rng.choice(n, size=int(0.03 * n), replace=False)
        level[nan_idx] = np.nan

    # Outliers (short spikes)
    if include_outliers:
        spike_idx = rng.choice(n, size=15, replace=False)
        level[spike_idx] += rng.choice([-3.0, 3.0], size=15)

    # Gaps (drop contiguous blocks of timesteps)
    if include_gaps:
        for _ in range(4):
            gap_start = rng.integers(0, n - 20)
            gap_len = rng.integers(5, 20)
            if gap_start + gap_len < n:
                time[gap_start:gap_start + gap_len] = time[gap_start - 1]  # duplicate → filtered by QC
                level[gap_start:gap_start + gap_len] = np.nan

    return time, level_clean, level


def make_toy_gauge(station_id: str = "S1", n_days: int = 30,
                   seed: int = 0, lat: float = 0.0, lon: float = 0.0,
                   include_storm: bool = True, include_gaps: bool = True,
                   include_outliers: bool = True, include_nans: bool = True,**kwargs) -> GaugeRecord:
    time, _, level = make_toy_tide_series(n_days=n_days, seed=seed,
                                             include_storm=include_storm,
                                             include_gaps=include_gaps,
                                             include_outliers=include_outliers,
                                             include_nans=include_nans)
    return GaugeRecord(
        station_id=station_id, units="m",
        sampling_period=timedelta(minutes=10),
        lat=lat, lon=lon,
        time=time, water_level=level,
        metadata={"toy": "synthetic"},
    )
