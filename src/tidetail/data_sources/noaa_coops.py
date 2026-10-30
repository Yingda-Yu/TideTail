"""NOAA CO-OPS API downloader.

Public, no auth, returns hourly water level observations. Verified endpoint
(2026-10-31):
    https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?station=8518750&begin_date=20240101&end_date=20240107&product=water_level&datum=MLLW&units=metric&time_zone=gmt&format=csv

NOTE: product MUST be "water_level" or "hourly_height", NOT "hourly"
("hourly" returns HTTP 400).

Station 8518750 = The Battery, New York City (hourly water level).
"""
from __future__ import annotations

import csv
from pathlib import Path


def build_url(station_id: str, begin_date: str, end_date: str,
              product: str = "water_level", datum: str = "MLLW",
              units: str = "metric", time_zone: str = "gmt",
              format_: str = "csv") -> str:
    """Build NOAA CO-OPS datagetter URL.

    Args:
        begin_date, end_date: YYYYMMDD format (e.g. "20240101")
        product: "hourly", "hilo", "daily", "maxsl"
        datum: "MLLW", "MLW", "MSL", "MHW", "MHHW"
    """
    base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    import urllib.parse
    params = {
        "station": station_id,
        "begin_date": begin_date,
        "end_date": end_date,
        "product": product,
        "datum": datum,
        "units": units,
        "time_zone": time_zone,
        "format": format_,
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def parse_csv(path: str | Path) -> list[dict[str, str]]:
    """Parse NOAA CO-OPS hourly CSV into list of {datetime, water_level, ...}."""
    rows: list[dict[str, str]] = []
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(dict(r))
    return rows


# Known NOAA stations relevant to TideTail regions:
KNOWN_STATIONS: dict[str, dict] = {
    "ny_bight_the_battery": {
        "station_id": "8518750", "region": "ny_bight",
        "lat": 40.7033, "lon": -74.0133,
        "description": "The Battery, New York City",
    },
    "thames_southend": {
        "station_id": "0160192", "region": "thames_north_sea",
        "lat": 51.51, "lon": 0.81,
        "description": "Southend-on-Sea, Thames Estuary",
        "provider": "UK Environment Agency / NOAA ID",
    },
    "southeast_australia_sydney": {
        "station_id": "AED146", "region": "southeast_australia",
        "description": "Sydney (BOM), note: NOAA uses different IDs for AU",
        "note": "Use BOM portal for direct AU data",
    },
    "us_gulf_coast_new_orleans": {
        "station_id": "8761724", "region": "us_gulf_coast",
        "lat": 29.8733, "lon": -90.0683,
        "description": "New Orleans, Lake Pontchartrain",
    },
    "south_atlantic_charleston": {
        "station_id": "8665530", "region": "south_atlantic_charleston",
        "lat": 32.7250, "lon": -79.9200,
        "description": "Charleston, SC",
    },
}
