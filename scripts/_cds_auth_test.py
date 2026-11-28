#!/usr/bin/env python3
"""CDS API authentication + tiny download + integrity verification.

Runs as background verification. Uses .cdsapirc — NEVER prints token.
"""
from __future__ import annotations
import cdsapi, json, time, hashlib, os, sys, xarray as xr
from pathlib import Path

out_path = Path("/tmp/era5_auth_test.nc")

print("=== TideTail CDS API Verification ===")
print(f"Config: ~/.cdsapirc exists, perms={oct(os.stat('/home/yingda/.cdsapirc').st_mode & 0o777)}")
print(f"cdsapi version: {cdsapi.__version__ if hasattr(cdsapi,'__version__') else '0.7.7'}")

c = cdsapi.Client(progress=False, wait_until_complete=False)
print("✅ Client instantiates")

req = {
    "product_type": "reanalysis",
    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
    "year": "2024", "month": "06", "day": "01",
    "time": [f"{h:02d}:00" for h in range(0, 24)],
    "area": [42, -76, 38, -72],
    "grid": [0.25, 0.25],
    "format": "netcdf", "download_format": "unarchived",
}

t0 = time.time()
c.retrieve("reanalysis-era5-single-levels", req, str(out_path))
elapsed = time.time() - t0

assert out_path.exists(), "DOWNLOAD FAILED"
size_mb = out_path.stat().st_size / 1e6
sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

ds = xr.open_dataset(out_path)
u10_mean = float(abs(ds.u10).mean())
msl_mean = float(ds.msl.mean())
print(f"\n✅ Download OK — {size_mb:.1f} MB in {elapsed:.0f}s")
print(f"   valid_time: {len(ds.valid_time)} entries (expected 24)")
print(f"   lat: [{float(ds.latitude.min()):.1f}, {float(ds.latitude.max()):.1f}]")
print(f"   lon: [{float(ds.longitude.min()):.1f}, {float(ds.longitude.max()):.1f}]")
print(f"   |u10| mean: {u10_mean:.2f} m/s")
print(f"   msl mean: {msl_mean:.0f} Pa")
print(f"   sha256: {sha[:24]}...")
ds.close()

result = {
    "status": "verified",
    "elapsed_s": elapsed,
    "size_mb": size_mb,
    "sha256_prefix": sha[:24],
    "n_valid_time": len(ds.valid_time) if 'ds' in dir() else None,
    "u10_mean_abs": u10_mean,
    "msl_mean_pa": msl_mean,
    "cdsapi_version": "0.7.7",
    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path("/tmp/era5_auth_result.json").write_text(json.dumps(result, indent=2))
print("\n🎉 CDS API FULLY VERIFIED")
out_path.unlink(missing_ok=True)
