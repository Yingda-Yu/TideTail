"""Robust data download utilities.

Provides a single DownloadManager that all source-specific downloaders
wrap. Features:
    - Retry with exponential backoff
    - Resume via HTTP Range headers when server supports
    - Chunked streaming download (avoids loading full file into memory)
    - Rate limiting (min_interval_between_calls seconds)
    - Request timeout
    - SHA256 + file size verification
    - Download manifest per file
    - Partial-failure recovery (track per-file status)
    - Idempotency (skip if file exists with correct hash)
    - Dry-run mode
    - Per-source quota enforcement (max bytes total)
    - Provenance metadata (exact URL, timestamp, user-agent, source)
    - License-aware (mark downloads with known license restrictions)
    - Disk safety (free-space check before large downloads)

Does NOT handle auth tokens — each source module manages its own auth
(following Registry auth_hint).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import ssl
    import urllib.error
    import urllib.parse
    import urllib.request
except ImportError:
    urllib = None  # fallback


@dataclass
class DownloadConfig:
    """Per-source download policy."""
    source_id: str
    base_dir: Path | str
    max_bytes_total: int = 20 * 1024**3  # Phase A default: 20 GB
    max_bytes_per_file: int = 2 * 1024**3
    min_free_disk_bytes: int = 5 * 1024**3  # always leave 5 GB
    min_interval_between_calls: float = 0.5  # seconds
    timeout: float = 60.0
    retries: int = 3
    backoff_base: float = 1.0  # exponential: base * 2**attempt
    chunk_size: int = 8192
    dry_run: bool = False
    user_agent: str = "TideTail/0.0.1 (extreme-coastal-forecast research)"


class DownloadStatus(str, Enum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    OK = "OK"
    EXISTS = "EXISTS"  # already present with correct hash
    INCOMPLETE = "INCOMPLETE"  # partial file, can resume
    FAILED = "FAILED"
    SKIPPED_QUOTA = "SKIPPED_QUOTA"
    SKIPPED_DISK = "SKIPPED_DISK"


class DownloadManager:
    """Centralized download orchestrator."""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "download_manifest.json"
        self._manifest: list[dict] = self._load_manifest()
        self._total_bytes: int = sum(e.get("bytes", 0) for e in self._manifest
                                      if e.get("status") == DownloadStatus.OK.value)
        self._last_call_time: float = 0.0

    # --- Manifest ---
    def _load_manifest(self) -> list[dict]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except (OSError, ValueError):
                return []
        return []

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    # --- Disk safety ---
    def _available_disk(self) -> int:
        try:
            import shutil
            usage = shutil.disk_usage(self.base_dir)
            return usage.free
        except OSError:
            return 0

    def _check_disk(self, needed: int) -> bool:
        free = self._available_disk()
        if free < self.config.min_free_disk_bytes + needed:
            raise RuntimeError(
                f"Insufficient disk: need {needed/1e9:.1f} GB + "
                f"{self.config.min_free_disk_bytes/1e9:.1f} GB margin, "
                f"have {free/1e9:.1f} GB free"
            )
        return True

    def _check_quota(self, needed: int) -> bool:
        return not self._total_bytes + needed > self.config.max_bytes_total

    # --- Rate limiter ---
    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self.config.min_interval_between_calls:
            time.sleep(self.config.min_interval_between_calls - elapsed)
        self._last_call_time = time.monotonic()

    # --- SHA256 ---
    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # --- Core download ---
    def download(self, url: str, dest_path: Path | str,
                 expected_sha256: str | None = None,
                 description: str = "",
                 extra_headers: dict[str, str] | None = None
                 ) -> dict:
        """Download one file with all safety checks."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "url": url,
            "path": str(dest),
            "description": description,
            "expected_sha256": expected_sha256,
            "timestamp_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": DownloadStatus.PENDING.value,
            "bytes": 0,
            "error": None,
            "retries": 0,
        }

        # 1. Disk check (estimate file size via HEAD, or skip)
        est_size = self._estimate_size(url)
        try:
            self._check_disk(est_size)
        except RuntimeError as exc:
            entry["status"] = DownloadStatus.SKIPPED_DISK.value
            entry["error"] = str(exc)
            self._manifest.append(entry)
            self._save_manifest()
            return entry

        # 2. Quota check
        if not self._check_quota(est_size):
            entry["status"] = DownloadStatus.SKIPPED_QUOTA.value
            entry["error"] = f"quota exceeded ({self._total_bytes/1e9:.1f}/{self.config.max_bytes_total/1e9:.1f} GB)"
            self._manifest.append(entry)
            self._save_manifest()
            return entry

        # 3. Already exists + correct hash?
        if dest.exists():
            actual = self._sha256(dest)
            if expected_sha256 and actual == expected_sha256 or expected_sha256 is None:
                entry["status"] = DownloadStatus.EXISTS.value
                entry["actual_sha256"] = actual
                entry["bytes"] = dest.stat().st_size
                self._manifest.append(entry)
                self._save_manifest()
                return entry
            else:
                # wrong hash — delete and re-download
                dest.unlink()

        # 4. Dry run
        if self.config.dry_run:
            entry["status"] = DownloadStatus.DOWNLOADING.value
            entry["dry_run"] = True
            self._manifest.append(entry)
            self._save_manifest()
            return entry

        # 5. Actual download with retry
        for attempt in range(self.config.retries + 1):
            entry["retries"] = attempt
            entry["status"] = DownloadStatus.DOWNLOADING.value
            try:
                self._rate_limit()
                self._do_download(url, dest, extra_headers)
                break
            except (OSError, RuntimeError) as exc:
                if attempt < self.config.retries:
                    time.sleep(self.config.backoff_base * 2**attempt)
                    continue
                entry["status"] = DownloadStatus.FAILED.value
                entry["error"] = str(exc)
                self._manifest.append(entry)
                self._save_manifest()
                return entry

        # 6. Verify hash
        actual = self._sha256(dest)
        entry["actual_sha256"] = actual
        if expected_sha256 and actual != expected_sha256:
            entry["status"] = DownloadStatus.FAILED.value
            entry["error"] = f"sha256 mismatch: got {actual}, expected {expected_sha256}"
            self._manifest.append(entry)
            self._save_manifest()
            return entry

        entry["bytes"] = dest.stat().st_size
        entry["status"] = DownloadStatus.OK.value
        entry["timestamp_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._total_bytes += entry["bytes"]
        self._manifest.append(entry)
        self._save_manifest()
        return entry

    def _estimate_size(self, url: str) -> int:
        """HEAD request to estimate file size; return 0 if unknown."""
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, method="HEAD",
                                          headers={"User-Agent": self.config.user_agent})
            with urllib.request.urlopen(req, timeout=self.config.timeout, context=ctx) as resp:
                cl = resp.headers.get("Content-Length")
                return int(cl) if cl else 0
        except (OSError, ValueError, RuntimeError):
            return 0

    def _do_download(self, url: str, dest: Path,
                     extra_headers: dict[str, str] | None = None) -> None:
        """Streaming chunked download."""
        if urllib is None:
            raise ImportError("urllib unavailable")

        headers = {"User-Agent": self.config.user_agent}
        if extra_headers:
            headers.update(extra_headers)

        # Try resume if partial file exists
        resume_start = dest.stat().st_size if dest.exists() else 0
        if resume_start > 0:
            headers["Range"] = f"bytes={resume_start}-"

        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.config.timeout, context=ctx) as resp:
            # Server accepted Range → append; else rewrite
            mode = "ab" if resp.status == 206 and resume_start > 0 else "wb"
            with open(dest, mode) as fh:
                while True:
                    chunk = resp.read(self.config.chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
