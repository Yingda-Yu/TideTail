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

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
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
        lines = [l.strip() for l in f]

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
        return datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
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
    # Normalize date_range bounds to match storm timezone awareness
    dr_start = dr_end = None
    if date_range is not None:
        first_storm = storms[0] if storms else None
        aware_tz = (
            first_storm.event_start_utc.tzinfo
            if first_storm is not None and first_storm.event_start_utc.tzinfo is not None
            else None
        )
        def _norm(d: datetime) -> datetime:
            if aware_tz is not None and d.tzinfo is None:
                return d.replace(tzinfo=aware_tz)
            return d
        dr_start = _norm(date_range[0])
        dr_end = _norm(date_range[1])

    for storm in storms:
        in_region = any(
            lat_range[0] <= tp.lat <= lat_range[1] and
            lng_range[0] <= tp.lng <= lng_range[1]
            for tp in storm.track
        )
        if not in_region:
            continue
        if dr_start is not None and (
            storm.event_end_utc < dr_start or storm.event_start_utc > dr_end
        ):
            continue
        result.append(storm)
    return result


# ---------------------------------------------------------------------------
# IBTrACS / HURDAT2 deduplication
# ---------------------------------------------------------------------------

def dedup_ibtracs_hurdat2(
    ibtracs_storms: list[Storm],
    hurdat2_storms: list[Storm],
    overlap_tol_hours: float = 48.0,
    overlap_tol_km: float = 300.0,
) -> list[Storm]:
    """Merge IBTrACS + HURDAT2, removing duplicate storms.

    IBTrACS uses SIDs like ``2016273N13300``. HURDAT2 uses IDs like ``AL152016``.
    Many Atlantic tropical cyclones appear in BOTH datasets. We keep the IBTrACS
    version (global coverage) but include HURDAT2 storms that are genuinely new.

    Dedup heuristic:
      1. Storm year + approximate centroid match (within overlap_tol_km) +
         overlapping time windows (within overlap_tol_hours).
      2. HURDAT2 storms are supplementary — include only if IBTrACS has no match.
    """
    merged = list(ibtracs_storms)
    ibtracs_years = {s.event_start_utc.year for s in ibtracs_storms}

    for h2 in hurdat2_storms:
        if h2.event_start_utc.year not in ibtracs_years:
            merged.append(h2)
            continue

        h2_center = (
            float(np.mean([tp.lat for tp in h2.track])),
            float(np.mean([tp.lng for tp in h2.track])),
        )
        h2_win_start = h2.event_start_utc - timedelta(hours=overlap_tol_hours)
        h2_win_end = h2.event_end_utc + timedelta(hours=overlap_tol_hours)

        is_dup = False
        for ib in ibtracs_storms:
            if ib.event_start_utc.year != h2.event_start_utc.year:
                continue
            ib_center = (
                float(np.mean([tp.lat for tp in ib.track])),
                float(np.mean([tp.lng for tp in ib.track])),
            )
            d_km = haversine_km(h2_center[0], h2_center[1], ib_center[0], ib_center[1])
            if d_km > overlap_tol_km:
                continue
            # Time overlap
            ib_end = ib.event_end_utc
            ib_start = ib.event_start_utc
            if ib_end >= h2_win_start and ib_start <= h2_win_end:
                is_dup = True
                break

        if not is_dup:
            merged.append(h2)

    return merged


# ---------------------------------------------------------------------------
# Event-wise split (critical for leakage prevention)
# ---------------------------------------------------------------------------

def build_canonical_event_groups(
    all_matches: list[StormGaugeMatch],
) -> dict[str, set[str]]:
    """Canonical storm → set of gauges it hit.

    ONE storm hitting 3 gauges is ONE canonical event, not three.
    Returns {canonical_storm_id: {station_id, ...}}.
    """
    groups: dict[str, set[str]] = {}
    for m in all_matches:
        groups.setdefault(m.canonical_storm_id, set()).add(m.station_id)
    return groups


def event_wise_split(
    canonical_events: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split canonical events into train/val/test by STORM ID.

    A single storm always stays in one split — no cross-split leakage
    even if it hit multiple gauges.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    ev = list(canonical_events)
    rng.shuffle(ev)

    n = len(ev)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    return {
        "train": ev[:n_train],
        "val": ev[n_train : n_train + n_val],
        "test": ev[n_train + n_val :],
    }


def verify_split_integrity(split_events: dict[str, list[str]]) -> dict[str, Any]:
    """Assert no storm appears in multiple splits. Returns diagnostics."""
    sets = {k: set(v) for k, v in split_events.items()}
    overlaps = {}
    split_names = list(sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            o = sets[split_names[i]] & sets[split_names[j]]
            overlaps[f"{split_names[i]}∩{split_names[j]}"] = len(o)

    total_unique = set()
    for s in sets.values():
        total_unique |= s

    all_clean = all(v == 0 for v in overlaps.values())

    return {
        "split_sizes": {k: len(v) for k, v in split_events.items()},
        "total_unique_events": len(total_unique),
        "overlaps": overlaps,
        "no_leakage": all_clean,
    }


def count_independent_storms(all_matches: list[StormGaugeMatch]) -> dict[str, int]:
    """Count TRUE independent storm events (one storm at any number of gauges = 1).

    CRITICAL: This is NOT len(all_matches). A single storm may appear in
    multiple gauge matches — that counts as 1 independent storm, not N.
    """
    canonical_ids = {m.canonical_storm_id for m in all_matches}
    gauge_storm_pairs = {(m.canonical_storm_id, m.station_id) for m in all_matches}

    return {
        "independent_storms": len(canonical_ids),
        "gauge_storm_pairs": len(gauge_storm_pairs),
        "total_matches": len(all_matches),
        "multi_gauge_storms": sum(
            1 for cid in canonical_ids
            if sum(1 for m in all_matches if m.canonical_storm_id == cid) > 1
        ),
    }


# ---------------------------------------------------------------------------
# Extratropical storm detection (placeholder — non-tropical Atlantic)
# ---------------------------------------------------------------------------

def mark_storm_type(
    storms: list[Storm],
    tropical_peak_wind_threshold: float = 34.0,  # knots (tropical storm)
) -> None:
    """Classify storm_type field in-place using peak wind + track latitude heuristic.

    Tropical cyclones have peak sustained wind ≥ 34 kt and form in tropics.
    Extratropical storms can be any wind speed but are poleward-moving.

    Note: True classification needs NHC metadata. This is a best-effort proxy.
    """
    for storm in storms:
        if storm.max_wind_kt is None:
            storm.storm_type = "unknown"
            continue
        peak_lat = max(tp.lat for tp in storm.track)
        if storm.max_wind_kt >= tropical_peak_wind_threshold and peak_lat < 40:
            storm.storm_type = "tropical_cyclone"
        elif storm.max_wind_kt >= tropical_peak_wind_threshold:
            storm.storm_type = "tropical_cyclone_late"
        else:
            storm.storm_type = "extratropical_or_subtropical"
