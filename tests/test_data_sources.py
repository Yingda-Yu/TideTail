"""Tests for src/tidetail/data_sources/ — registry, download manager, NOAA CO-OPS.

All tests are CPU-only, no network, no real downloads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tidetail.data_sources import (
    REGISTRY,
    DataSource,
    DownloadConfig,
    DownloadManager,
    DownloadStatus,
    SourceStatus,
    priority_sources,
    registry_to_dicts,
    save_registry,
)
from tidetail.data_sources.noaa_coops import KNOWN_STATIONS, build_url, parse_csv

# ==========================================================================
# Registry tests
# ==========================================================================

def test_registry_not_empty():
    assert len(REGISTRY) >= 10, f"Expected >=10 sources, got {len(REGISTRY)}"


def test_registry_all_data_sources():
    for s in REGISTRY:
        assert isinstance(s, DataSource)


def test_registry_required_fields():
    """Every registered source must have source_id, provider, url, status."""
    required = ["source_id", "provider", "dataset_name", "url", "retrieval_status"]
    for s in REGISTRY:
        for fname in required:
            val = getattr(s, fname)
            assert val, f"Source {s.source_id!r} missing field {fname!r}"


def test_source_status_enum():
    expected = {
        "DISCOVERED", "VERIFIED", "LICENSE_CHECKED", "DOWNLOADED",
        "VALIDATED", "ADMITTED", "EXCLUDED", "AUTH_REQUIRED",
    }
    actual = {m.name for m in SourceStatus}
    assert actual == expected


def test_priority_sources_only_priority_1():
    p1 = priority_sources()
    assert len(p1) > 0, "Expected at least one priority-1 source"
    for s in p1:
        assert s.priority == 1


def test_priority_sources_sorted_by_then_source_id():
    p1 = priority_sources()
    ids = [(s.retrieval_status, s.source_id) for s in p1]
    assert ids == sorted(ids)


def test_registry_to_dicts_roundtrip():
    dicts = registry_to_dicts()
    assert len(dicts) == len(REGISTRY)
    for d in dicts:
        assert "source_id" in d
        assert "provider" in d


def test_save_registry_writes_valid_json(tmp_path: Path):
    out = tmp_path / "catalog" / "sources.json"
    save_registry(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == len(REGISTRY)


def test_verified_sources_in_registry():
    """NOAA CO-OPS and HURDAT2 must be VERIFIED."""
    ids = {s.source_id: s for s in REGISTRY}
    assert ids["noaa_coops"].retrieval_status == SourceStatus.VERIFIED.value
    assert ids["noaa_hurdat2"].retrieval_status == SourceStatus.VERIFIED.value


# ==========================================================================
# DownloadManager tests (dry-run + local file, no network)
# ==========================================================================

def test_download_config_defaults(tmp_path: Path):
    cfg = DownloadConfig(source_id="test", base_dir=tmp_path)
    assert cfg.max_bytes_total == 20 * 1024**3
    assert cfg.retries == 3
    assert cfg.chunk_size == 8192


def test_download_manager_init_creates_dir(tmp_path: Path):
    base = tmp_path / "dl"
    dm = DownloadManager(DownloadConfig(source_id="test", base_dir=base))
    assert base.is_dir()
    assert dm.manifest_path == base / "download_manifest.json"


def test_sha256_static_method(tmp_path: Path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert DownloadManager._sha256(f) == expected


def test_check_quota_allows_within_budget(tmp_path: Path):
    dm = DownloadManager(DownloadConfig(
        source_id="test", base_dir=tmp_path,
        max_bytes_total=1024,
    ))
    # Initially _total_bytes = 0
    assert dm._check_quota(500) is True
    assert dm._check_quota(1024) is True
    assert dm._check_quota(1025) is False


def test_check_quota_respects_existing_bytes(tmp_path: Path):
    dm = DownloadManager(DownloadConfig(
        source_id="test", base_dir=tmp_path,
        max_bytes_total=1024,
    ))
    dm._total_bytes = 900
    assert dm._check_quota(200) is False   # 900 + 200 > 1024
    assert dm._check_quota(100) is True    # 900 + 100 <= 1024


def test_check_disk_raises_when_insufficient(tmp_path: Path, monkeypatch):
    """Mock disk_usage so we control free space regardless of real FS."""
    import shutil
    from types import SimpleNamespace

    dm = DownloadManager(DownloadConfig(
        source_id="test", base_dir=tmp_path,
        min_free_disk_bytes=10 * 1024**3,
    ))
    fake_usage = SimpleNamespace(total=100 * 1024**3, used=99 * 1024**3,
                                 free=1 * 1024**3)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: fake_usage)
    with pytest.raises(RuntimeError):
        dm._check_disk(1024)


def test_dry_run_does_not_create_file(tmp_path: Path):
    dm = DownloadManager(DownloadConfig(
        source_id="test", base_dir=tmp_path, dry_run=True,
    ))
    dest = tmp_path / "data.txt"
    entry = dm.download(
        url="https://example.com/nonexistent",
        dest_path=dest,
        description="test dry run",
    )
    assert entry["status"] == DownloadStatus.DOWNLOADING.value
    assert "dry_run" in entry
    assert not dest.exists()


def test_manifest_saved_after_dry_run(tmp_path: Path):
    dm = DownloadManager(DownloadConfig(
        source_id="test", base_dir=tmp_path, dry_run=True,
    ))
    dm.download(url="https://example.com/a", dest_path=tmp_path / "a.txt")
    dm.download(url="https://example.com/b", dest_path=tmp_path / "b.txt")
    manifest = json.loads(dm.manifest_path.read_text())
    assert len(manifest) == 2


def test_download_status_enum():
    expected = {
        "PENDING", "DOWNLOADING", "OK", "EXISTS", "INCOMPLETE",
        "FAILED", "SKIPPED_QUOTA", "SKIPPED_DISK",
    }
    actual = {m.name for m in DownloadStatus}
    assert actual == expected


def test_quota_skip_status(tmp_path: Path):
    dm = DownloadManager(DownloadConfig(
        source_id="test", base_dir=tmp_path,
        max_bytes_total=1,  # trivially small
    ))
    # Force _total_bytes so quota is exceeded even before size estimate
    dm._total_bytes = 100
    entry = dm.download(
        url="https://example.com/file.bin",
        dest_path=tmp_path / "file.bin",
    )
    assert entry["status"] == DownloadStatus.SKIPPED_QUOTA.value


# ==========================================================================
# NOAA CO-OPS tests
# ==========================================================================

def test_build_url_default_product_is_water_level():
    url = build_url("8518750", "20240101", "20240107")
    assert "product=water_level" in url


def test_build_url_contains_expected_params():
    url = build_url("8518750", "20240101", "20240107",
                   product="water_level", datum="MLLW")
    assert "station=8518750" in url
    assert "begin_date=20240101" in url
    assert "end_date=20240107" in url
    assert "datum=MLLW" in url
    assert "format=csv" in url


def test_build_url_does_not_use_hourly_product():
    """Regression: 'hourly' param returns HTTP 400 from NOAA."""
    url = build_url("8518750", "20240101", "20240107")
    assert "product=hourly&" not in url
    assert "product=water_level" in url


def test_known_stations_has_five_regions():
    regions = {v["region"] for v in KNOWN_STATIONS.values()}
    expected = {"ny_bight", "thames_north_sea", "southeast_australia",
                "us_gulf_coast", "south_atlantic_charleston"}
    assert regions == expected


def test_known_stations_entries_have_station_id():
    for name, info in KNOWN_STATIONS.items():
        assert "station_id" in info, f"{name} missing station_id"
        assert info["station_id"], f"{name} station_id is empty"


def test_parse_csv(tmp_path: Path):
    csv_text = (
        "Time,Water Level,Sigma,O, F, R, L, Quality\n"
        "2024-01-01 00:00,1.234,0.010,0,0,0,0,v\n"
        "2024-01-01 01:00,1.567,0.012,0,0,0,0,v\n"
    )
    p = tmp_path / "noaa.csv"
    p.write_text(csv_text)
    rows = parse_csv(p)
    assert len(rows) == 2
    assert rows[0]["Time"] == "2024-01-01 00:00"
    assert rows[0]["Water Level"] == "1.234"
