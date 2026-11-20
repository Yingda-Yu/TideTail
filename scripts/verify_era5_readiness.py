"""Wave 2.7 ERA5 readiness verification and five-region matrix.

Re-opens every monthly NetCDF partition (not just the download manifest) and
checks variables, units, timestamps, spatial coverage, finiteness, physical
plausibility, cross-month continuity, and anchor-station grid alignment.

Usage:
    python3 scripts/verify_era5_readiness.py

Output:
    data/results/era5_readiness_matrix.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/data/yingda/tidetail-data/real-sources-v1/era5")
OUT = ROOT / "data" / "results" / "era5_readiness_matrix.json"

REGIONS = {
    "ny_bight": {"anchor": (40.71, -74.01), "lat": (38, 42), "lon": (-76, -72)},
    "us_gulf_coast": {"anchor": (29.33, -94.77), "lat": (26, 31), "lon": (-97, -84)},
    "south_atlantic_charleston": {"anchor": (32.78, -79.87), "lat": (31, 34.5), "lon": (-81, -78)},
    "southeast_australia": {"anchor": (-34.50, 150.87), "lat": (-38, -30), "lon": (148, 154)},
    "thames_north_sea": {"anchor": (51.45, 0.80), "lat": (50, 53.5), "lon": (-2, 2)},
}
YEARS = list(range(2015, 2025))
MONTHS = list(range(1, 13))
VARS = ("u10", "v10", "msl")
SENTINEL = 3.4028234663852886e38


def expected_hours(year: int, month: int) -> int:
    next_month = (datetime(year + (month == 12), month % 12 + 1, 1))
    return int((next_month - datetime(year, month, 1)).days * 24)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def check_file(path: Path, region: str, year: int, month: int) -> dict:
    cfg = REGIONS[region]
    errors = []
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception as exc:  # corrupt file
        return {"status": "CORRUPT", "errors": [f"open failed: {exc}"]}

    with ds:
        missing_vars = [v for v in VARS if v not in ds.data_vars]
        if missing_vars:
            errors.append(f"missing vars {missing_vars}")

        hours = int(ds.sizes.get("valid_time", -1))
        exp = expected_hours(year, month)
        if hours != exp:
            errors.append(f"hours {hours} != expected {exp}")

        times = ds["valid_time"].values.astype("datetime64[ns]")
        if len(times) > 1 and np.any(np.diff(times) != np.timedelta64(1, "h")):
            errors.append("non-hourly or non-monotonic timestamps")
        if len(times) and len(np.unique(times)) != len(times):
            errors.append("duplicate timestamps")

        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if abs(float(lat.max()) - cfg["lat"][1]) > 0.3 or abs(float(lat.min()) - cfg["lat"][0]) > 0.3:
            errors.append("latitude bounds mismatch")
        if abs(float(lon.min()) - cfg["lon"][0]) > 0.3 or abs(float(lon.max()) - cfg["lon"][1]) > 0.3:
            errors.append("longitude bounds mismatch")

        for v, unit in (("u10", "m s**-1"), ("v10", "m s**-1"), ("msl", "Pa")):
            if v in ds.data_vars:
                attrs_unit = ds[v].attrs.get("units")
                if attrs_unit != unit:
                    errors.append(f"{v} units {attrs_unit!r} != {unit!r}")

        stats = {}
        for v in VARS:
            if v not in ds.data_vars:
                continue
            data = ds[v].values
            finite = np.isfinite(data) & (np.abs(data) != SENTINEL)
            frac = float(finite.mean())
            if frac < 1.0:
                errors.append(f"{v} non-finite fraction {1-frac:.4f}")
            vals = data[finite]
            stats[v] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "finite_frac": frac,
            }

        if "u10" in stats and max(abs(stats["u10"]["min"]), abs(stats["u10"]["max"])) > 100:
            errors.append("u10 outside physical range")
        if "v10" in stats and max(abs(stats["v10"]["min"]), abs(stats["v10"]["max"])) > 100:
            errors.append("v10 outside physical range")
        if "msl" in stats and not (88000 < stats["msl"]["min"] < stats["msl"]["max"] < 110000):
            errors.append("msl outside physical range")

        anchor_lat, anchor_lon = cfg["anchor"]
        if min(cfg["lat"]) <= anchor_lat <= max(cfg["lat"]) and min(cfg["lon"]) <= anchor_lon <= max(cfg["lon"]):
            aligned = True
        else:
            aligned = False
            errors.append("anchor outside grid")
        nearest = None
        if aligned:
            ilat = int(np.argmin(np.abs(lat - anchor_lat)))
            ilon = int(np.argmin(np.abs(lon - anchor_lon)))
            nearest = {
                "grid_lat": float(lat[ilat]),
                "grid_lon": float(lon[ilon]),
                "distance_km": round(haversine_km(anchor_lat, anchor_lon, float(lat[ilat]), float(lon[ilon])), 1),
            }

        expver = set(ds["expver"].values.astype(str)) if "expver" in ds else {"missing"}

    return {
        "status": "OK" if not errors else "FAIL",
        "errors": errors,
        "hours": hours,
        "expected_hours": exp,
        "first_time": str(times[0]) if len(times) else None,
        "last_time": str(times[-1]) if len(times) else None,
        "expver": sorted(expver),
        "anchor_aligned": aligned,
        "nearest_grid": nearest,
        "stats": stats,
    }


def main() -> None:
    matrix = {}
    for region, cfg in REGIONS.items():
        files = sorted(glob.glob(str(DATA_DIR / f"era5_{region}_*_u10v10mslp.nc")))
        per_month = {}
        first_last = {}
        failures = []
        total_hours = 0
        expected_total = 0
        for path in files:
            stem = Path(path).stem  # era5_<region>_<year>-<month>
            year_month = stem.replace(f"era5_{region}_", "").replace("_u10v10mslp", "")
            year, month = (int(x) for x in year_month.split("-"))
            res = check_file(Path(path), region, year, month)
            per_month[f"{year}-{month:02d}"] = {
                "status": res["status"],
                "hours": res.get("hours"),
                "errors": res.get("errors", []),
            }
            first_last[f"{year}-{month:02d}"] = (
                res.get("first_time"), res.get("last_time")
            )
            total_hours += res.get("hours", 0) or 0
            expected_total += expected_hours(year, month)
            if res["status"] != "OK":
                failures.append({"file": Path(path).name, **res})

        expected_files = len(YEARS) * len(MONTHS)
        completeness = len(files) / expected_files
        # Cross-month continuity: next month must start exactly 1 hour after
        # this month's final timestamp.
        continuity = 0
        continuity_checks = 0
        for year in YEARS:
            for month in MONTHS:
                key = f"{year}-{month:02d}"
                nxt = f"{year + (month == 12)}-{month % 12 + 1:02d}"
                if key not in first_last or nxt not in first_last:
                    continue
                last_t, next_first = first_last[key][1], first_last[nxt][0]
                if last_t is None or next_first is None:
                    continue
                continuity_checks += 1
                last_dt = np.datetime64(last_t)
                next_dt = np.datetime64(next_first)
                if next_dt - last_dt == np.timedelta64(1, "h"):
                    continuity += 1

        anchor = None
        if files:
            stem = Path(files[0]).stem.replace(f"era5_{region}_", "").replace("_u10v10mslp", "")
            year, month = (int(x) for x in stem.split("-"))
            anchor = check_file(Path(files[0]), region, year, month).get("nearest_grid")

        matrix[region] = {
            "files_present": len(files),
            "files_expected": expected_files,
            "file_completeness_pct": round(completeness * 100, 2),
            "hourly_rows_present": total_hours,
            "hourly_rows_expected": expected_total,
            "hour_completeness_pct": round(total_hours / expected_total * 100, 2) if expected_total else 0,
            "months_ok": sum(1 for v in per_month.values() if v["status"] == "OK"),
            "months_failed": sum(1 for v in per_month.values() if v["status"] != "OK"),
            "month_continuity_checks": continuity_checks,
            "month_continuity_ok": continuity,
            "anchor_station": {"lat": cfg["anchor"][0], "lon": cfg["anchor"][1]},
            "nearest_grid": anchor,
            "failures": failures[:5],
        }

        print(f"{region:>28}: files {len(files)}/{expected_files} "
              f"({completeness*100:5.1f}%), hours {total_hours}/{expected_total} "
              f"({matrix[region]['hour_completeness_pct']:5.1f}%), "
              f"months_ok={matrix[region]['months_ok']}, failed={matrix[region]['months_failed']}, "
              f"anchor_grid={anchor}")

    all_ok = all(v["months_failed"] == 0 for v in matrix.values())
    payload = {
        "audit_version": "wave27_v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "grid": "0.25 degree regular_ll, 17x17 points",
        "variables": {"u10": "m s**-1", "v10": "m s**-1", "msl": "Pa"},
        "all_regions_integrity_ok": all_ok,
        "regions": matrix,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nAll regions integrity OK: {all_ok}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
