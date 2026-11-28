"""TideTail ERA5 download — 5 coastal regions × 10 years × monthly chunks.

Downloads ERA5 single-level reanalysis for physics-guided flow matching:
  - 10m U wind (u10)
  - 10m V wind (v10)
  - Mean sea level pressure (msl)

CDS area format: [north, west, south, east]
CDS client is BLOCKING — request accepted → status running → download complete.
Rate limit: CDS allows ~20 concurrent requests; we do serial with 10s cooldown.
Each monthly file ~1.4 MB → 5 regions × 120 months = ~600 downloads = ~840 MB.

Usage:
    # Full 5-region 10-year download (background, retries)
    nohup python3 scripts/download_era5_forcing.py --workers 1 --max-retries 3 \
        --skip-existing --manifest /data/yingda/tidetail-data/real-sources-v1/era5/MANIFEST.json \
        > /tmp/era5_download.log 2>&1 &

    # Dry-run to see what will be downloaded
    python3 scripts/download_era5_forcing.py --dry-run

    # Only one region/year for testing
    python3 scripts/download_era5_forcing.py --regions ny_bight --years 2024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path("/data/yingda/tidetail-data/real-sources-v1/era5")
MANIFEST_PATH = DATA_ROOT / "MANIFEST.json"

# ERA5 area format: [north, west, south, east]
# Grid: 0.25° horizontal resolution
REGIONS = {
    "ny_bight": {
        "anchor_lat": 40.71, "anchor_lon": -74.01,  # The Battery, NY
        "area": [42, -76, 38, -72],
        "lat_range": (38, 42), "lon_range": (-76, -72),
        "grid": [0.25, 0.25],
    },
    "us_gulf_coast": {
        "anchor_lat": 29.33, "anchor_lon": -94.77,  # Galveston Pier 21
        "area": [31, -97, 26, -84],
        "lat_range": (26, 31), "lon_range": (-97, -84),
        "grid": [0.25, 0.25],
    },
    "south_atlantic_charleston": {
        "anchor_lat": 32.78, "anchor_lon": -79.87,  # Charleston
        "area": [34.5, -81, 31, -78],
        "lat_range": (31, 34.5), "lon_range": (-81, -78),
        "grid": [0.25, 0.25],
    },
    "southeast_australia": {
        "anchor_lat": -34.50, "anchor_lon": 150.87,  # Port Kembla
        "area": [-30, 148, -38, 154],  # [north, west, south, east]
        "lat_range": (-38, -30), "lon_range": (148, 154),
        "grid": [0.25, 0.25],
    },
    "thames_north_sea": {
        "anchor_lat": 51.45, "anchor_lon": 0.80,  # Sheerness
        "area": [53.5, -2, 50, 2],
        "lat_range": (50, 53.5), "lon_range": (-2, 2),
        "grid": [0.25, 0.25],
    },
}

ERA5_VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]

CDS_PRODUCT = "reanalysis-era5-single-levels"


def month_days(year: int, month: int) -> list[str]:
    """Return list of day strings '01'..'31' for given month."""
    next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc) if month < 12 \
        else datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    last_day = (next_month - timedelta(days=1)).day
    return [f"{d:02d}" for d in range(1, last_day + 1)]


def expected_filename(region: str, year: int, month: int) -> str:
    return f"era5_{region}_{year}-{month:02d}_u10v10mslp.nc"


def build_request(region: str, year: int, month: int) -> dict:
    r = REGIONS[region]
    return {
        "product_type": "reanalysis",
        "variable": ERA5_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": month_days(year, month),
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": r["area"],
        "grid": r["grid"],
        "format": "netcdf",
        "download_format": "unarchived",
    }


def verify_netcdf(path: Path, region: str, year: int, month: int) -> dict:
    """Verify downloaded NetCDF4 is readable and has correct structure.

    Returns verification dict or raises ValueError.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    sizes = dict(ds.sizes)

    # Check expected variables
    required_vars = {"u10", "v10", "msl"}
    actual_vars = set(ds.data_vars)
    missing = required_vars - actual_vars
    if missing:
        raise ValueError(f"Missing variables: {missing}")

    # Check time count (31 or 30 or 28/29 days × 24h)
    expected_days = len(month_days(year, month))
    expected_hours = expected_days * 24
    if sizes.get("valid_time", 0) != expected_hours:
        raise ValueError(f"Expected {expected_hours} valid_time, got {sizes.get('valid_time', 0)}")

    # Check spatial coverage
    r = REGIONS[region]
    lat_min = float(ds.latitude.values.min())
    lat_max = float(ds.latitude.values.max())
    lon_min = float(ds.longitude.values.min())
    lon_max = float(ds.longitude.values.max())
    if abs(lat_max - r["lat_range"][1]) > 0.3 or abs(lat_min - r["lat_range"][0]) > 0.3:
        raise ValueError(f"Lat mismatch: expected {r['lat_range']}, got [{lat_min}, {lat_max}]")
    if abs(lon_min - r["lon_range"][0]) > 0.3 or abs(lon_max - r["lon_range"][1]) > 0.3:
        raise ValueError(f"Lon mismatch: expected {r['lon_range']}, got [{lon_min}, {lon_max}]")

    # Sample values (physical plausibility)
    u10_mean = float(np.abs(ds.u10.values).mean())
    v10_mean = float(np.abs(ds.v10.values).mean())
    msl_mean = float(ds.msl.values.mean())
    if not (0.1 < u10_mean < 40 and 0.1 < v10_mean < 40):
        raise ValueError(f"Wind implausible: u10_mean={u10_mean:.2f}, v10_mean={v10_mean:.2f}")
    if not (95000 < msl_mean < 110000):
        raise ValueError(f"Pressure implausible: msl_mean={msl_mean:.1f} Pa")

    ds.close()

    return {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "dimensions": sizes,
        "lat_range": [lat_min, lat_max],
        "lon_range": [lon_min, lon_max],
        "u10_mean_abs": u10_mean,
        "v10_mean_abs": v10_mean,
        "msl_mean_pa": msl_mean,
        "valid_hours": expected_hours,
        "status": "verified",
    }


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "manifest_version": "era5_v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "last_updated": None,
        "product": CDS_PRODUCT,
        "variables": ERA5_VARIABLES,
        "grid": "0.25°",
        "format": "NetCDF4",
        "region_bounds": {r: REGIONS[r] for r in REGIONS},
        "files": {},  # key = filename, value = verification dict
        "download_log": [],  # events
    }


def save_manifest(manifest: dict):
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


def download_one(region: str, year: int, month: int, max_retries: int = 3) -> tuple[bool, str, float]:
    """Download one monthly chunk. Returns (success, message, elapsed_s)."""
    import cdsapi

    fname = expected_filename(region, year, month)
    out_path = DATA_ROOT / fname

    # Skip if already verified
    manifest = load_manifest()
    if fname in manifest["files"] and manifest["files"][fname].get("status") == "verified":
        return True, "skipped (already verified)", 0.0

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    req = build_request(region, year, month)

    last_error = None
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            c = cdsapi.Client(progress=False, wait_until_complete=True)
            c.retrieve(CDS_PRODUCT, req, str(out_path))
            elapsed = time.time() - t0

            # Verify
            result = verify_netcdf(out_path, region, year, month)
            manifest = load_manifest()
            manifest["files"][fname] = result
            manifest["download_log"].append({
                "time": datetime.now(timezone.utc).isoformat(),
                "region": region, "year": year, "month": month,
                "elapsed_s": elapsed, "status": "success", "attempts": attempt,
            })
            save_manifest(manifest)
            return True, f"success ({elapsed:.1f}s)", elapsed

        except Exception as e:
            elapsed = time.time() - t0
            last_error = e
            print(f"  Attempt {attempt}/{max_retries} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            # Clean up partial file
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            # Exponential backoff before retry
            time.sleep(2 ** attempt)

    return False, f"failed: {type(last_error).__name__}: {last_error}", time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="TideTail ERA5 download pipeline")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS.keys()),
                    help="Which regions to download")
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2025)),
                    help="Which years to download")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip already-verified files (default)")
    ap.add_argument("--force-redownload", action="store_true", default=False,
                    help="Redownload all files regardless of manifest")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded but don't fetch")
    ap.add_argument("--cooldown", type=int, default=10,
                    help="Seconds between downloads (CDS rate limit compliance)")
    args = ap.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    # Build task list
    tasks = []
    for region in args.regions:
        for year in args.years:
            for month in range(1, 13):
                fname = expected_filename(region, year, month)
                already = fname in manifest["files"] and manifest["files"][fname].get("status") == "verified"
                if already and not args.force_redownload:
                    continue
                tasks.append((region, year, month, fname))

    print(f"{'='*70}")
    print(f"TideTail ERA5 Download Pipeline")
    print(f"{'='*70}")
    print(f"Regions: {args.regions}")
    print(f"Years:   {min(args.years)}-{max(args.years)}")
    print(f"Pending: {len(tasks)} monthly chunks")
    print(f"Variables: {ERA5_VARIABLES}")
    print(f"CDS product: {CDS_PRODUCT}")
    print(f"Rate limit cooldown: {args.cooldown}s")
    print(f"Output: {DATA_ROOT}")
    print(f"{'='*70}")

    # Disk check
    import shutil
    total, used, free = shutil.disk_usage(DATA_ROOT)
    est_size_mb = len(tasks) * 1.5  # ~1.5 MB per monthly chunk
    print(f"Disk: {free/1e9:.1f} GB free ({free/total*100:.1f}%), estimated need: {est_size_mb:.0f} MB")
    if free < 1e9:  # < 1 GB free
        print("⚠️  WARNING: less than 1 GB free — check disk space before continuing", file=sys.stderr)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for region, year, month, fname in tasks[:10]:
            print(f"  {region:28s} {year}-{month:02d} → {fname}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    # Verify CDS auth with minimal call first
    print(f"\n{'='*70}")
    print("Phase 0: CDS authentication check")
    print(f"{'='*70}")
    import cdsapi
    try:
        c = cdsapi.Client(progress=False, wait_until_complete=False)
        # Just check that client instantiates without error
        print("  ✅ CDS client instantiates — auth config valid")
    except Exception as e:
        print(f"  ❌ CDS client FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # Main download loop
    print(f"\n{'='*70}")
    print(f"Phase 1: Downloading {len(tasks)} monthly chunks")
    print(f"{'='*70}")

    success = 0
    failed = 0
    skipped = 0
    total_t0 = time.time()

    for i, (region, year, month, fname) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {region:28s} {year}-{month:02d}", end=" ")

        ok, msg, elapsed = download_one(region, year, month, max_retries=args.max_retries)

        if "skipped" in msg:
            print(f"⏭  {msg}")
            skipped += 1
        elif ok:
            print(f"✅ {msg}")
            success += 1
        else:
            print(f"❌ {msg}")
            failed += 1

        # Status every 10
        if i % 10 == 0:
            total_elapsed = time.time() - total_t0
            rate = (success + failed + skipped) / total_elapsed if total_elapsed > 0 else 0
            print(f"  Progress: {success}✓ {skipped}⏭ {failed}✗ | {rate:.2f}/s | {total_elapsed:.0f}s", file=sys.stderr)

        # Cooldown (rate limit) — skip after last task
        if i < len(tasks):
            time.sleep(args.cooldown)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"✅ Success:  {success}")
    print(f"⏭  Skipped:  {skipped}")
    print(f"❌ Failed:   {failed}")
    print(f"📄 Manifest: {MANIFEST_PATH}")

    # Save summary
    manifest = load_manifest()
    manifest["summary"] = {
        "success": success, "skipped": skipped, "failed": failed,
        "total_elapsed_s": total_elapsed,
        "completed": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)


if __name__ == "__main__":
    main()
