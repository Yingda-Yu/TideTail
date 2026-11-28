"""TideTail data_sources subpackage — global coastal data registry + downloaders.

Available:
    registry            — DataSource registry, status enum, JSON export
    download_manager    — Robust download manager (retry, resume, sha256, quota)
    noaa_coops          — NOAA CO-OPS public API downloader (no auth needed)
"""
from .download_manager import DownloadConfig, DownloadManager, DownloadStatus
from .registry import (
    REGISTRY,
    DataSource,
    SourceStatus,
    priority_sources,
    registry_to_dicts,
    save_registry,
)

__all__ = [
    "REGISTRY",
    "DataSource",
    "DownloadConfig",
    "DownloadManager",
    "DownloadStatus",
    "SourceStatus",
    "priority_sources",
    "registry_to_dicts",
    "save_registry",
]
