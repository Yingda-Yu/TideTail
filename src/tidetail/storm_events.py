"""Storm-event database for TideTail.

Loads IBTrACS/HURDAT2 tropical cyclone tracks and matches them
to NOAA gauge windows by distance, time, and intensity.

Canonical schema per storm-event:
    canonical_storm_id:  str            # e.g. "ibtracs_2024320N35278"
    storm_type:          str            # "tropical_cyclone" | "extratropical" | ...
    source_dataset:      str            # "IBTrACS" | "HURDAT2"
    source_storm_id:     str            # original ID in source
    event_start_utc:     datetime       # first track point
    event_end_utc:       datetime       # last track point
    max_wind_kt:         float          # max sustained wind (knots)
    min_pressure_mb:     float          # min central pressure (hPa)
    track_points:         list[dict]     # lat, lng, time, wind, pres
    matched_gauges:      list[dict]     # station_id, nearest_track_dist_km, time_offset_hours
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# IBTrACS loader
# ---------------------------------------------------------------------------

def load_ibtracs_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load IBTrACS v04r01 CSV — skips the two-line header.

    IBTrACS CSVs have TWO header rows:
        row 0: column names
        row 1: units ("degrees_north", "kts", "mb", ...)
        rows 2+: data
    """
    path = Path(path)
    with open(path, "r") as f:
        all_lines = f.readlines()

    header = all_lines[0].strip().split(",")
    # Skip row 1 (units) — it's a data row that says "degrees_north" etc.
    data_lines = all_lines[2:]

    rows: list[dict[str, Any]] = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 10:
            continue
        row = dict(zip(header, parts))
        rows.append(row)
    return rows


def load_hurdat2(path: str | Path) -> list[dict[str, Any]]:
    """Load HURDAT2 North Atlantic / Eastern Pacific CSV.

    HURDAT2 format: first row per storm is header (AL/SC + year + name),
    then data rows with 6-hour track points.
    """
    path = Path(path)
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    rows: list[dict[str, Any]] = []
    storm_id = ""
    for line in lines:
        if not line:
            continue
        parts = line.split(",")
        if len(parts) == 3 and parts[0].startswith(("AL", "SC", "CP", "EP")):
            storm_id = parts[0].strip()
            continue
        if len(parts) < 5:
            continue
        rows.append({
            "HURDAT2_ID": storm_id,
            "ISO_TIME": parts[0].strip(),
            "LAT": parts[2].strip(),
            "LON": parts[3].strip(),
            "WIND_SPEED": parts[4].strip() if len(parts) > 4 else "",
            "PRES": parts[5].strip() if len(parts) > 5 else "",
        })
    return rows


# ---------------------------------------------------------------------------
# Canonical storm representation
# ---------------------------------------------------------------------------

@dataclass
class TrackPoint:
    lat: float
    lng: float
    time: datetime
    wind_kt: float | None = None
    pressure_mb: float | None = None


@dataclass
class Storm:
    canonical_storm_id: str
    storm_type: str
    source_dataset: str
    source_storm_id: str
    event_start_utc: datetime
    event_end_utc: datetime
    max_wind_kt: float | None = None
    min_pressure_mb: float | None = None
    track: list[TrackPoint] = field(default_factory=list)

    def duration_hours(self) -> float:
        return (self.event_end_utc - self.event_start_utc).total_seconds() / 3600

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_start_utc"] = self.event_start_utc.isoformat()
        d["event_end_utc"] = self.event_end_utc.isoformat()
        for tp in d["track"]:
            tp["time"] = tp["time"].isoformat()
        return d


# ---------------------------------------------------------------------------
# Parsers: raw rows → Storm objects
# ---------------------------------------------------------------------------

def _parse_iso_time(s: str) -> datetime | None:
    """IBTrACS ISO_TIME looks like '2024-10-25 12:00:00'."""
    s = s.strip()
    if not s:
        return None
    try:
        return datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _parse_lat_lon(lat_str: str, lon_str: str) -> tuple[float, float] | None:
    """IBTrACS uses LAT (degrees_north, negative = south) and LON (degrees_east,
    sometimes > 180 = east of prime meridian beyond dateline).
    """
    try:
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
    except (ValueError, AttributeError):
        return None
    # Convert longitudes > 180 to -180..180
    if lon > 180:
        lon -= 360
    return lat, lon


def parse_ibtracs_to_storms(rows: list[dict[str, Any]]) -> list[Storm]:
    """Group IBTrACS rows by SID and build Storm objects."""
    storms: dict[str, list[TrackPoint]] = {}
    wind_max: dict[str, float] = {}
    pres_min: dict[str, float] = {}

    for r in rows:
        sid = r.get("SID", "").strip()
        if not sid or len(sid) < 8:
            continue
        parsed = _parse_lat_lon(r.get("LAT", ""), r.get("LON", ""))
        if parsed is None:
            continue
        lat, lng = parsed
        t = _parse_iso_time(r.get("ISO_TIME", ""))
        if t is None:
            continue

        wind = None
        wstr = r.get("WMO_WIND", "").strip()
        if wstr and wstr not in ("", " ", "degrees_north", "kts"):
            try:
                wind = float(wstr)
            except ValueError:
                pass

        pres = None
        pstr = r.get("WMO_PRES", "").strip()
        if pstr and pstr not in ("", " ", "mb"):
            try:
                pres = float(pstr)
            except ValueError:
                pass

        tp = TrackPoint(lat=lat, lng=lng, time=t, wind_kt=wind, pressure_mb=pres)
        storms.setdefault(sid, []).append(tp)

        if wind is not None:
            wind_max[sid] = max(wind_max.get(sid, 0.0), wind)
        if pres is not None and pres > 800:  # sanity check
            pres_min[sid] = min(pres_min.get(sid, 1200.0), pres)

    result: list[Storm] = []
    for sid, track in storms.items():
        track.sort(key=lambda x: x.time)
        if not track:
            continue
        result.append(Storm(
            canonical_storm_id=f"ibtracs_{sid}",
            storm_type="tropical_cyclone",
            source_dataset="IBTrACS",
            source_storm_id=sid,
            event_start_utc=track[0].time,
            event_end_utc=track[-1].time,
            max_wind_kt=wind_max.get(sid),
            min_pressure_mb=pres_min.get(sid),
            track=track,
        ))
    return result


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Storm ↔ gauge matching
# ---------------------------------------------------------------------------

@dataclass
class GaugeStation:
    station_id: str
    lat: float
    lng: float
    name: str = ""


@dataclass
class StormGaugeMatch:
    canonical_storm_id: str
    station_id: str
    nearest_track_dist_km: float
    time_offset_hours: float  # |peak_time - gauge_window_center|
    storm_peak_wind_kt: float | None


def match_storms_to_gauges(
    storms: list[Storm],
    gauge: GaugeStation,
    max_dist_km: float = 500.0,
) -> list[StormGaugeMatch]:
    """Find all storms whose track comes within max_dist_km of gauge.

    Uses the storm's nearest track point distance and the time offset
    between storm peak (max wind) and the nearest track point time.
    """
    matches: list[StormGaugeMatch] = []
    for storm in storms:
        best_dist = float("inf")
        best_tp: TrackPoint | None = None

        for tp in storm.track:
            d = haversine_km(gauge.lat, gauge.lng, tp.lat, tp.lng)
            if d < best_dist:
                best_dist = d
                best_tp = tp

        if best_dist > max_dist_km or best_tp is None:
            continue

        # Storm peak time: track point with max wind
        peak_tp = max(storm.track, key=lambda x: x.wind_kt or 0.0)
        time_offset = abs((peak_tp.time - best_tp.time).total_seconds()) / 3600

        matches.append(StormGaugeMatch(
            canonical_storm_id=storm.canonical_storm_id,
            station_id=gauge.station_id,
            nearest_track_dist_km=round(best_dist, 1),
            time_offset_hours=round(time_offset, 1),
            storm_peak_wind_kt=storm.max_wind_kt,
        ))

    return matches


# ---------------------------------------------------------------------------
# Regional storm catalog — precompute for TideTail regions
# ---------------------------------------------------------------------------

def storms_in_region(
    storms: list[Storm],
    lat_range: tuple[float, float],
    lng_range: tuple[float, float],
    date_range: tuple[datetime, datetime] | None = None,
) -> list[Storm]:
    """Filter storms whose track enters a lat/lon bounding box.

    longitudes are expected in -180..180 range.
    """
    result = []
    for storm in storms:
        in_region = any(
            lat_range[0] <= tp.lat <= lat_range[1] and
            lng_range[0] <= tp.lng <= lng_range[1]
            for tp in storm.track
        )
        if not in_region:
            continue
        if date_range is not None:
            if storm.event_end_utc < date_range[0] or storm.event_start_utc > date_range[1]:
                continue
        result.append(storm)
    return result
