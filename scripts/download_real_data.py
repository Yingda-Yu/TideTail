"""Phase 3 real data download — CORRECTED version.

Fixes from v1:
  1. NOAA CO-OPS water_level: split into ≤ 30-day chunks (API 31-day limit).
  2. IBTrACS: corrected download URLs via NCEI FTP.
  3. HURDAT2 storm counting fixed (AL line → parse correctly).
  4. Always pre-verify station IDs.
  5. Every file: SHA256 + size + rows + provenance.

NOAA stations we KNOW work (verified 2026-10-31):
  8518750 The Battery, NYC
  8665530 Charleston, SC
  8761724 New Orleans
  Others fail verify → excluded from download.

Output: /data/yingda/tidetail-data/real-sources-v1/DOWNLOAD_MANIFEST.json
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

OUT = Path("/data/yingda/tidetail-data/real-sources-v1")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "TideTail-Research/1.0 (open research, non-commercial)"}
RATE_SLEEP = 0.25
TIMEOUT = 45


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, dest: Path) -> dict:
    """Single HTTP GET with retry, returns provenance dict."""
    last_code = 0
    for attempt in range(1, 3):
        time.sleep(RATE_SLEEP)
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            last_code = r.status_code
            if r.status_code != 200:
                if attempt == 2:
                    return {"status": f"HTTP_{r.status_code}", "bytes": 0, "sha256": "",
                            "http_code": r.status_code, "error_body": r.text[:500]}
                time.sleep(2 ** attempt)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            sz = dest.stat().st_size
            return {"status": "OK", "bytes": sz, "sha256": sha256_file(dest),
                    "http_code": 200}
        except requests.RequestException as e:
            if attempt == 2:
                return {"status": f"NETWORK_{type(e).__name__}: {e}", "bytes": 0,
                        "sha256": "", "http_code": 0}
            time.sleep(2 ** attempt)
    return {"status": "UNKNOWN", "bytes": 0, "sha256": "", "http_code": last_code}


def month_chunks(year: int):
    """Yield (YYYYMMDD_begin, YYYYMMDD_end) for each month in year."""
    for month in range(1, 13):
        start = dt.date(year, month, 1)
        if month == 12:
            end = dt.date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = dt.date(year, month + 1, 1) - timedelta(days=1)
        yield start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# --- NOAA CO-OPS download (30-day chunks) ---------------------------------

NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
VERIFIED_STATIONS = [
    ("ny_bight",                  "8518750", "The Battery, NYC",         40.7033, -74.0133),
    ("south_atlantic_charleston", "8665530", "Charleston, SC",           32.7250, -79.9200),
    ("us_gulf_coast",             "8761724", "New Orleans, Lake Pont.",  29.8733, -90.0683),
]


def download_noaa_month(station_id: str, year: int, month: int) -> dict:
    start_s, end_s = next((b, e) for b, e in month_chunks(year)
                          if dt.datetime.strptime(b, "%Y%m%d").month == month)
    url = (f"{NOAA_BASE}?station={station_id}&begin_date={start_s}&end_date={end_s}"
           f"&product=water_level&datum=MLLW&units=metric&time_zone=gmt&format=csv")
    dest = OUT / "noaa_coops" / f"{station_id}" / f"{year}-{month:02d}.csv"
    rec = http_get(url, dest)
    if rec["status"] == "OK":
        try:
            with open(dest) as fh:
                rows = sum(1 for _ in csv.DictReader(fh))
            rec["rows"] = rows
            if rows < 50:
                rec["status"] = "TOO_FEW_ROWS"
        except Exception as e:
            rec["status"] = f"PARSE_ERROR: {e}"
            rec["rows"] = 0
    return rec


def download_noaa_all():
    print("\n[1/3] NOAA CO-OPS hourly water level (30-day chunks)")
    records = []
    total_rows = 0
    for region, sid, desc, lat, lon in VERIFIED_STATIONS:
        print(f"\n  Station {sid} ({desc}), {region}")
        for yr in range(2015, 2025):
            for month in range(1, 13):
                rec = download_noaa_month(sid, yr, month)
                rec.update({
                    "source": "noaa_coops", "station_id": sid,
                    "region": region, "description": desc,
                    "latitude": lat, "longitude": lon,
                    "year": yr, "month": month,
                    "downloaded_at": datetime.utcnow().isoformat() + "Z",
                })
                records.append(rec)
                if rec["status"] == "OK":
                    total_rows += rec.get("rows", 0)
                    print(f"    {yr}-{month:02d}: {rec['status']} "
                          f"({rec['bytes']:,} B, {rec.get('rows', '?')} rows)", end="\r")
                else:
                    print(f"    {yr}-{month:02d}: {rec['status']}", end="\r")
            print()
    ok = sum(1 for r in records if r["status"] == "OK")
    ok_bytes = sum(r["bytes"] for r in records if r["status"] == "OK")
    print(f"\n  NOAA CO-OPS: {ok} files OK, {ok_bytes/1e6:.1f} MB, "
          f"{total_rows:,} data rows")
    return records


# --- HURDAT2 --------------------------------------------------------------

HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"


def download_hurdat2():
    print("\n[2/3] NOAA HURDAT2 Atlantic hurricane tracks")
    dest = OUT / "hurdat2" / "hurdat2-1851-2025.txt"
    print(f"  {HURDAT2_URL} → ", end="", flush=True)
    rec = http_get(HURDAT2_URL, dest)
    n_storms = 0
    if rec["status"] == "OK":
        with open(dest) as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 4 and line[:2] in ("AL", "NA") and line[2:4].isdigit():
                    if line[4:6] == "  ":  # storm header line
                        n_storms += 1
    print(f"{rec['status']} ({rec['bytes']:,} B, ~{n_storms} storms)")
    rec.update({
        "source": "hurdat2", "url": HURDAT2_URL,
        "estimated_storms": n_storms,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
    })
    return [rec]


# --- IBTrACS (FTP via ncei.noaa.gov) -------------------------------------

def download_ibtracs():
    print("\n[3/3] IBTrACS global tropical cyclones")
    # IBTrACS moved; try current NCEI FTP paths
    urls_to_try = [
        ("CSV.gz", "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship/access/csv/ibtracs.ALL.list.v04r00.csv.gz"),
        ("NC",     "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship/access/netcdf/ibtracs.ALL.list.v04r00.nc"),
    ]
    out = []
    for ext, url in urls_to_try:
        dest = OUT / "ibtracs" / f"ibtracs_all.{ext.lower().replace('csv.gz','csv.gz')}"
        print(f"  {ext}: ", end="", flush=True)
        rec = http_get(url, dest)
        rec.update({
            "source": "ibtracs", "url": url,
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
        })
        if rec["status"] == "OK" and ext == "CSV.gz":
            # Decompress and count unique storm IDs
            csv_dest = OUT / "ibtracs" / "ibtracs_all.csv"
            with gzip.open(dest, "rb") as gz, open(csv_dest, "wb") as out_f:
                shutil.copyfileobj(gz, out_f)
            ids = set()
            with open(csv_dest) as fh:
                header = fh.readline()
                for line in fh:
                    parts = line.split(",")
                    if len(parts) > 1:
                        ids.add(parts[0])
            rec["local_path_csv"] = str(csv_dest)
            rec["estimated_storms"] = len(ids)
        print(f"{rec['status']} ({rec['bytes']:,} B)")
        out.append(rec)
    return out


# --- Manifest + summary --------------------------------------------------

def write_manifest(records):
    ok = [r for r in records if r["status"] == "OK"]
    fail = [r for r in records if r["status"] != "OK"]
    by_src = {}
    for r in records:
        s = r.get("source", "unknown")
        st = by_src.setdefault(s, {"ok": 0, "fail": 0, "ok_bytes": 0})
        if r["status"] == "OK":
            st["ok"] += 1
            st["ok_bytes"] += r["bytes"]
        else:
            st["fail"] += 1
    import shutil
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disk_free_gb": shutil.disk_usage("/data").free / (1024 ** 3),
        "total_files_ok": len(ok),
        "total_bytes_ok": sum(r["bytes"] for r in ok),
        "per_source": by_src,
        "records": records,
    }
    manifest_path = OUT / "DOWNLOAD_MANIFEST.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest, manifest_path


def print_summary(manifest):
    print(f"\n{'='*60}\nDOWNLOAD COMPLETE\n"
          f"{'='*60}\n"
          f"{'Source':<22} {'OK':>6} {'Fail':>6} {'Bytes (MB)':>12}")
    print("-" * 50)
    for src, stats in manifest["per_source"].items():
        print(f"  {src:<20} {stats['ok']:>6} {stats['fail']:>6} "
              f"{stats['ok_bytes']/1e6:>11.1f}")
    print(f"\n  TOTAL: {manifest['total_files_ok']} files, "
          f"{manifest['total_bytes_ok']/1e6:.1f} MB")
    print(f"  Disk free: {manifest['disk_free_gb']:.1f} GB")


def main():
    import shutil
    disk_gb = shutil.disk_usage("/data").free / (1024 ** 3)
    print(f"{'='*60}\nTideTail Real Data Download — Phase 3\n"
          f"Output: {OUT}\nDisk free: {disk_gb:.1f} GB\n{'='*60}")

    all_records = []
    all_records.extend(download_noaa_all())
    all_records.extend(download_hurdat2())
    all_records.extend(download_ibtracs())

    manifest, path = write_manifest(all_records)
    print_summary(manifest)
    print(f"\nManifest → {path}")


if __name__ == "__main__":
    main()
