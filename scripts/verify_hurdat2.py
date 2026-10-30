"""Verify NOAA HURDAT2 Atlantic hurricane data endpoint.

Public, no auth. Parses the official TXT format. This is the canonical
Atlantic tropical cyclone track dataset.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_n] = "2"

import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tidetail.data_sources import DownloadConfig, DownloadManager

URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"


def main():
    print("=== HURDAT2 Validation ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="tidetail_hurdat_"))
    mgr = DownloadManager(DownloadConfig(source_id="hurdat2", base_dir=tmpdir))

    dest = tmpdir / "hurdat2.txt"
    result = mgr.download(URL, dest, description="HURDAT2 Atlantic hurricanes 1850-2023")
    print(f"Download: {result['status']}, {result.get('bytes', 0)} bytes")

    if result["status"] != "OK":
        print(f"ERROR: {result.get('error')}")
        sys.exit(1)

    # Quick parse: HURDAT2 has header lines starting with "AL" + digit
    text = dest.read_text(errors="replace")
    lines = text.strip().split("\n")
    header_count = 0
    total_tracks = 0
    for ln in lines:
        if ln.startswith("AL") and not ln.startswith("HURDAT"):
            header_count += 1
            parts = ln.split(",")
            if len(parts) >= 3:
                parts[1].strip()
                parts[2].strip()
            # Count data lines until next header
        elif ln.strip() and not ln.startswith("AL") and header_count > 0:
            total_tracks += 1

    print(f"Header lines (storms): {header_count}")
    print(f"Data rows: {total_tracks}")
    print(f"Sample header: {lines[0][:80]}")
    print(f"SHA256: {result.get('actual_sha256')}")
    print("=== SUCCESS ===")


if __name__ == "__main__":
    main()
