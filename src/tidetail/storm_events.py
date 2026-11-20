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
    matched_gauges:      list[dict]     # station_id, nearest_track_dist_km,
                                         # storm_arrival_to_gauge_peak_hours

Validity tiers (separate questions, NOT a single hierarchy):
    TRACK_MATCHED:        Storm track comes within distance threshold of gauge
    OBSERVATION_MATCHED:  Gauge has hourly water-level data overlapping
                          the storm's track period
    WATER_LEVEL_EVENT_VALIDATED:  Observed water level shows genuine
                          surge response concurrent with storm
    FORMAL_TEST_ADMITTED: Storm appears in the test split AND its windows
                          feed the forecasting pipeline
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# IBTrACS loader — uses csv module, handles quoted fields correctly
# ---------------------------------------------------------------------------

def load_ibtracs_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load IBTrACS v04r01 CSV — skips the two-line header.

    Uses Python's csv module for standards-compliant parsing (handles
    quoted fields, escaped commas correctly). Naive split(",") worked
    for the specific v04r01 file (no quoted fields observed in 312K rows)
    but the csv module is required for correctness.

    IBTrACS CSVs have TWO header rows:
        row 0: column names (174 columns)
        row 1: units ("degrees_north", "kts", "mb", ...)
        rows 2+: data
    """
    path = Path(path)
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # row 0: column names
        # Skip row 1 (units)
        next(reader)

        rows: list[dict[str, Any]] = []
        for row in reader:
            if not row or len(row) < 10:
                continue
            # Pad short rows
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            # Only zip up to header length
            rows.append(dict(zip(header, row[:len(header)])))

    return rows


def load_hurdat2(path: str | Path) -> list[dict[str, Any]]:
    """Load HURDAT2 North Atlantic / Eastern Pacific / IO / Western Pacific.

    HURDAT2 format (critical differences from IBTrACS):
      - First row per storm: header with 3 CSV fields
          AL011851,            UNNAMED,       14,
          ^basin+num+year      ^name          ^n_data_rows
      - Data rows use YYYYMMDD (no hyphens!) + HHMM (no colons!)
          18510625, 0000,  , HU, 28.0N,  94.8W,  80, ...
          col 0      col 1  col 2 col 3 col 4    col 5    col 6
          date_utc   time_utc (blank?) status lat_str  lon_str  max_wind_kt
      - Latitude has N/S suffix (e.g. "28.0N")
      - Longitude has E/W suffix (e.g. "94.8W" → negative, "120.0E" → positive)
    """
    path = Path(path)
    with open(path, "r") as f:
        content = f.read()

    rows: list[dict[str, Any]] = []
    storm_id = ""
    storm_name = ""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # Storm header: 3 parts, first part starts with basin prefix + numeric
        if (
            len(parts) >= 3
            and len(parts[0]) >= 6
            and parts[0][:2] in ("AL", "SC", "CP", "EP", "IO", "WP")
            and parts[0][2:4].isdigit()
        ):
            storm_id = parts[0].strip()
            storm_name = parts[1].strip() if len(parts) > 1 else ""
            continue

        # Data row
        if len(parts) < 5:
            continue

        raw_date = parts[0].strip()  # YYYYMMDD
        raw_time = parts[1].strip()  # HHMM
        status = parts[3].strip() if len(parts) > 3 else ""
        lat_str = parts[4].strip() if len(parts) > 4 else ""
        lon_str = parts[5].strip() if len(parts) > 5 else ""
        wind_str = parts[6].strip() if len(parts) > 6 else ""

        # Parse HURDAT2-specific lat/lon format (28.0N, 94.8W)
        lat = None
        lng = None
        if lat_str and lat_str[-1] in ("N", "S"):
            try:
                lat = float(lat_str[:-1])
                if lat_str[-1] == "S":
                    lat = -lat
            except ValueError:
                lat = None

        if lon_str and lon_str[-1] in ("E", "W"):
            try:
                lng = float(lon_str[:-1])
                if lon_str[-1] == "W":
                    lng = -lng
            except ValueError:
                lng = None

        # Parse HURDAT2 date/time (YYYYMMDD + HHMM, no separators)
        dt = None
        if len(raw_date) == 8 and len(raw_time) >= 3:
            try:
                y, mo, d = int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8])
                hh = int(raw_time[:2])
                mm = int(raw_time[2:4]) if len(raw_time) >= 4 else 0
                dt = datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)
            except (ValueError, IndexError):
                dt = None

        wind = None
        if wind_str and wind_str not in ("-999", "-99", ""):
            try:
                w = float(wind_str)
                if 0 < w < 200:  # sanity
                    wind = w
            except ValueError:
                pass

        rows.append({
            "HURDAT2_ID": storm_id,
            "HURDAT2_NAME": storm_name,
            "ISO_TIME": dt.isoformat() if dt else "",
            "STATUS": status,
            "LAT": str(lat) if lat is not None else "",
            "LON": str(lng) if lng is not None else "",
            "WIND_SPEED": str(wind) if wind is not None else "",
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


def parse_hurdat2_to_storms(rows: list[dict[str, Any]]) -> list[Storm]:
    """Group HURDAT2 rows by HURDAT2_ID and build Storm objects.

    Uses the STATUS field (HU, TS, ET, etc.) for better storm typing
    than a pure latitude+wind heuristic.
    """
    storms: dict[str, list[TrackPoint]] = {}
    wind_max: dict[str, float] = {}
    status_set: dict[str, set[str]] = {}

    for r in rows:
        hid = r.get("HURDAT2_ID", "").strip()
        if not hid:
            continue

        lat_str = r.get("LAT", "").strip()
        lng_str = r.get("LON", "").strip()
        try:
            lat = float(lat_str) if lat_str else None
            lng = float(lng_str) if lng_str else None
        except ValueError:
            lat = lng = None
        if lat is None or lng is None:
            continue

        t_str = r.get("ISO_TIME", "").strip()
        t = None
        if t_str:
            try:
                t = datetime.fromisoformat(t_str)
            except ValueError:
                pass
        if t is None:
            continue

        wind = None
        wstr = r.get("WIND_SPEED", "").strip()
        if wstr:
            try:
                w = float(wstr)
                if 0 < w < 200:
                    wind = w
            except ValueError:
                pass

        tp = TrackPoint(lat=lat, lng=lng, time=t, wind_kt=wind)
        storms.setdefault(hid, []).append(tp)
        status_set.setdefault(hid, set()).add(r.get("STATUS", "").strip())

        if wind is not None:
            wind_max[hid] = max(wind_max.get(hid, 0.0), wind)

    def _classify(hid: str) -> str:
        statuses = status_set.get(hid, set())
        winds = wind_max.get(hid)
        if "ET" in statuses:
            return "extratropical_or_subtropical"
        if any(s in statuses for s in ("HU", "TY", "TC", "CY")):
            return "tropical_cyclone"
        if any(s in statuses for s in ("TS", "TD")):
            return "tropical_depression_or_storm"
        if winds is not None and winds >= 64:
            return "tropical_cyclone"
        return "unknown"

    result: list[Storm] = []
    for hid, track in storms.items():
        track.sort(key=lambda x: x.time)
        if not track:
            continue
        result.append(Storm(
            canonical_storm_id=f"hurdat2_{hid}",
            storm_type=_classify(hid),
            source_dataset="HURDAT2",
            source_storm_id=hid,
            event_start_utc=track[0].time,
            event_end_utc=track[-1].time,
            max_wind_kt=wind_max.get(hid),
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
    storm_peak_to_nearest_track_hours: float  # |storm_peak_time - nearest_track_point_time|
    storm_peak_wind_kt: float | None
    validity_tier: str = "TRACK_MATCHED"  # one of TRACK_MATCHED, OBSERVATION_MATCHED, etc.


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
        peak_to_track = abs((peak_tp.time - best_tp.time).total_seconds()) / 3600

        matches.append(StormGaugeMatch(
            canonical_storm_id=storm.canonical_storm_id,
            station_id=gauge.station_id,
            nearest_track_dist_km=round(best_dist, 1),
            storm_peak_to_nearest_track_hours=round(peak_to_track, 1),
            storm_peak_wind_kt=storm.max_wind_kt,
            validity_tier="TRACK_MATCHED",
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
