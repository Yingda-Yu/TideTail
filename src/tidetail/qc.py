"""Quality-control primitives for water-level observations.

QC pipeline:
  1. timestamp monotonicity + duplicate detection
  2. gap and missing-rate reporting
  3. outlier masking (3σ + IQR)
  4. physical-bounds clamp (e.g. [-5, 15] m, configurable per-gauge)
  5. NaN linear interpolation up to gap threshold
  6. QC summary report
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from .data import GaugeRecord, missing_ratio, outlier_3sigma, outlier_iqr

PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    # site_code -> (lo, hi) in metres; defaults covered by DEFAULT
    "_default": (-5.0, 15.0),
}


@dataclass
class QCSummary:
    station_id: str
    n_total: int = 0
    n_valid: int = 0
    n_duplicates: int = 0
    n_monotonicity_violations: int = 0
    missing_fraction: float = 0.0
    n_outliers_3sigma: int = 0
    n_outliers_iqr: int = 0
    n_bounds_clamped: int = 0
    bounds_applied: tuple[float, float] = (0.0, 0.0)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


def _bounds_for(station_id: str) -> tuple[float, float]:
    return PHYSICAL_BOUNDS.get(station_id, PHYSICAL_BOUNDS["_default"])


def qc_pipeline(rec: GaugeRecord,
                expected_period: timedelta | None = None,
                bounds: tuple[float, float] | None = None,
                max_interp_gap_s: float = 3600.0,
                ) -> tuple[GaugeRecord, QCSummary]:
    """Run full QC on a GaugeRecord. Returns cleaned copy + summary.

    Pipeline steps run in order — earlier masks aren't re-used by later steps,
    so outlier detection sees the post-bounds-clamp series.
    """
    expected_period = expected_period or rec.sampling_period
    bounds = bounds or _bounds_for(rec.station_id)

    out = GaugeRecord(
        station_id=rec.station_id, units=rec.units,
        sampling_period=rec.sampling_period, lat=rec.lat, lon=rec.lon,
        time=rec.time.copy(), water_level=rec.water_level.copy(),
        metadata=dict(rec.metadata),
    )
    summary = QCSummary(station_id=rec.station_id, bounds_applied=bounds)
    summary.n_total = out.n

    if out.n == 0:
        summary.notes.append("empty record")
        return out, summary

    # 1. Duplicates
    dup_mask = np.zeros(out.n, dtype=bool)
    dup_mask[1:] = out.time[1:] == out.time[:-1]
    summary.n_duplicates = int(dup_mask.sum())
    if dup_mask.any():
        keep = ~dup_mask
        out.time = out.time[keep]
        out.water_level = out.water_level[keep]

    # 2. Monotonicity
    mono_violations = int((np.diff(out.time.astype("datetime64[s]")) <= np.timedelta64(0, "s")).sum())
    summary.n_monotonicity_violations = mono_violations
    if mono_violations:
        # drop non-monotonic entries
        diff = np.diff(out.time.astype("datetime64[s]"))
        ok = np.concatenate([[True], diff > np.timedelta64(0, "s")])
        out.time = out.time[ok]
        out.water_level = out.water_level[ok]

    # 3. Missing ratio & gaps
    summary.missing_fraction = missing_ratio(out.time, expected_period)

    # 4. Physical bounds clamp (set OOB → NaN so outlier detection ignores them)
    lo, hi = bounds
    bad = (out.water_level < lo) | (out.water_level > hi)
    summary.n_bounds_clamped = int(bad.sum())
    out.water_level[bad] = np.nan

    # 5. Outliers
    o3 = outlier_3sigma(out.water_level)
    summary.n_outliers_3sigma = int(o3.sum())
    outlier_iqr_mask = outlier_iqr(out.water_level)
    summary.n_outliers_iqr = int(outlier_iqr_mask.sum())
    out.water_level[o3 | outlier_iqr_mask] = np.nan

    # 6. Interpolate short gaps
    if out.n >= 2:
        t_s = (out.time.astype("datetime64[s]") - out.time[0].astype("datetime64[s]"))
        t_s = t_s.astype("timedelta64[s]").astype(float)
        from .data import linear_interpolate  # local import to avoid cycle
        out.water_level = linear_interpolate(out.water_level, max_gap_s=max_interp_gap_s, time_s=t_s)

    summary.n_valid = int(np.isfinite(out.water_level).sum())
    summary.notes.append(f"keep {summary.n_valid}/{summary.n_total} "
                         f"(missing {summary.missing_fraction:.1%}, "
                         f"clamped {summary.n_bounds_clamped}, "
                         f"outliers {summary.n_outliers_3sigma + summary.n_outliers_iqr})")
    return out, summary
