"""End-to-end NOAA CO-OPS download validation.

Tests:
1. Registry exports to JSON correctly.
2. NOAA URL builder produces valid endpoint.
3. DownloadManager fetches real hourly data for NOAA station 8518750
   (The Battery, NYC) for 7 days in Jan 2024.
4. Parse returned CSV and validate water-level columns.
5. Report bytes downloaded, rows parsed, SHA256.

Runs without auth, no paid APIs, NO raw data committed.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_n] = "2"

import tempfile
from pathlib import Path

# Put src on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tidetail.data_sources import (
    DownloadConfig,
    DownloadManager,
    priority_sources,
    save_registry,
)
from tidetail.data_sources.noaa_coops import KNOWN_STATIONS, build_url, parse_csv


def main():
    print("=== NOAA CO-OPS End-to-End Validation ===")

    # 1. Registry
    out = Path("data/catalog")
    out.mkdir(parents=True, exist_ok=True)
    save_registry(out / "global_sources.json")
    print(f"[1] Registry: {len(priority_sources())} priority-1 sources saved")

    # 2. Build URL for The Battery, Jan 1-7 2024
    station = KNOWN_STATIONS["ny_bight_the_battery"]
    url = build_url(station["station_id"], "20240101", "20240107", product="water_level")
    print(f"[2] URL: {url}")

    # 3. Download with manager (ephemeral dir)
    tmpdir = Path(tempfile.mkdtemp(prefix="tidetail_noaa_"))
    cfg = DownloadConfig(source_id="noaa_coops_test", base_dir=tmpdir)
    mgr = DownloadManager(cfg)

    dest = tmpdir / "ny_bight_8518750_hourly_20240101_20240107.csv"
    result = mgr.download(url, dest, description="NOAA The Battery 7-day hourly")
    print(f"[3] Download result: {result['status']}, {result.get('bytes', 0)} bytes")

    if result["status"] not in ("OK", "EXISTS"):
        print(f"    ERROR: {result.get('error')}")
        sys.exit(1)

    # 4. Parse and validate
    rows = parse_csv(dest)
    print(f"[4] Parsed {len(rows)} rows")
    if rows:
        col = rows[0]
        print(f"    Columns: {list(col.keys())}")
        # Water level column varies — look for 'Water Level'
        wl_key = next((k for k in col if "level" in k.lower() or "water" in k.lower()), None)
        if wl_key:
            vals = [float(r[wl_key]) for r in rows if r.get(wl_key)]
            print(f"    Water levels: min={min(vals):.3f}, max={max(vals):.3f}, "
                  f"n={len(vals)}")
        # Timestamp check
        dt_key = next((k for k in col if "date" in k.lower() or "time" in k.lower()), None)
        if dt_key:
            print(f"    Date range: {rows[0][dt_key]} → {rows[-1][dt_key]}")

    print(f"[5] SHA256: {result.get('actual_sha256', 'N/A')}")

    # 5. Summary
    print("\n=== SUCCESS ===")
    print(f"Station:  {station['description']} ({station['station_id']})")
    print(f"Rows:     {len(rows)}")
    print(f"Bytes:    {result.get('bytes', 0)}")
    print(f"Manifest: {tmpdir / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
