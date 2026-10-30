"""Read only the authorized owner copy; prepare train-only engineering windows."""

import csv
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np

OWNER_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024")


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def safe_file(root, relative):
    path = root / relative
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError(f"refusing symlink/non-file: {relative}")
    return path


def train_end_with_embargo(cut, events, embargo=96):
    """Half-open train row block; purge entire threshold intervals touching boundary."""
    end = cut - embargo
    while True:
        crossing = [start for start, stop in events if start < end <= stop]
        if not crossing:
            return end
        end = min(crossing)


def windows_from_train(values, context=72, horizon=24, stride=1):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < context + horizon or not np.isfinite(values).all():
        raise ValueError("insufficient/nonfinite training rows; no imputation")
    mean, std = float(values.mean()), float(values.std())
    if std <= 0:
        raise ValueError("constant training series")
    normalized = ((values - mean) / std).astype("float32")
    windows = np.lib.stride_tricks.sliding_window_view(normalized, context + horizon)[::stride]
    return windows[:, :context].copy(), windows[:, context:].copy(), mean, std


def prepare(root=OWNER_ROOT):
    root = Path(root).absolute()
    if root != OWNER_ROOT or root.resolve() != root:
        raise ValueError("only the authorized owner copy is allowed")
    hashes = {}

    def checked(relative, expected=None):
        path = safe_file(root, relative)
        hashes[relative] = digest(path)
        if expected is not None and hashes[relative] != expected:
            raise ValueError(f"hash mismatch: {relative}")
        return path

    inventory = json.loads(checked("file_inventory_sha256.json").read_text())
    entries = {entry["path"]: entry for entry in inventory["files"]}

    def release_file(relative):
        entry = entries[relative]
        path = checked(relative, entry["sha256"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("release size mismatch")
        return path

    migration = safe_file(root.parent, "MIGRATION_MANIFEST.json")
    migration_data = json.loads(migration.read_text())
    if migration_data["destination"] != str(root):
        raise ValueError("migration destination mismatch")
    migration_entries = {entry["path"]: entry for entry in migration_data["file_entries"]}
    hashes["../MIGRATION_MANIFEST.json"] = digest(migration)
    availability_path = checked("region_availability.json")
    migration_entry = migration_entries["region_availability.json"]
    if hashes["region_availability.json"] != migration_entry["sha256"]:
        raise ValueError("admission metadata does not match migration hash")
    availability = json.loads(availability_path.read_text())
    region = next(r for r in availability["regions"] if r["region"] == "ny_bight")
    if region["formal_training_authorized"] is not True:
        raise ValueError("region is not admitted")
    prefix = "derived/prime_coast/ny_bight/"
    manifest_path = release_file(prefix + "region_manifest_v1.json")
    manifest = json.loads(manifest_path.read_text())
    if (
        not manifest["formal_training_authorized"]
        or digest(manifest_path) != region["region_manifest_sha256"]
    ):
        raise ValueError("region manifest/admission mismatch")
    target_path = release_file(prefix + "target_timeline_v1.csv")
    event_path = release_file(prefix + "event_table_v1.csv")
    with target_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != manifest["rows"]:
        raise ValueError("row count mismatch")
    times = [datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")) for row in rows]
    if any((b - a).total_seconds() != 3600 for a, b in itertools.pairwise(times)):
        raise ValueError("non-hourly target timestamps")
    stations = sorted({row["anchor_id"] for row in rows})
    if len(stations) != 1:
        raise ValueError("pilot requires exactly one station")
    with event_path.open() as handle:
        event_rows = list(csv.DictReader(handle))
    intervals = []
    ids = set()
    for event in event_rows:
        start, stop = int(event["start_index"]), int(event["end_index"])
        if not 0 <= start <= stop < len(rows) or event["event_id"] in ids:
            raise ValueError("invalid/duplicate event interval")
        ids.add(event["event_id"])
        intervals.append((start, stop))
    ordered = sorted(intervals)
    if any(a[1] >= b[0] for a, b in itertools.pairwise(ordered)):
        raise ValueError("overlapping threshold event intervals")
    cut = int(0.8 * len(rows))
    end = train_end_with_embargo(cut, intervals)
    train_rows = rows[:end]  # split/purge BEFORE normalization or windows
    if any(row["is_missing"] != "0" for row in train_rows):
        raise ValueError("missing training observation")
    residual = np.array([float(row["water_level_m"]) for row in train_rows])
    total = np.array([float(row["observed_total_water_level_m"]) for row in train_rows])
    if not np.isfinite(total).all():
        raise ValueError("nonfinite total water")
    history, target, mean, std = windows_from_train(residual)
    train_ids = [e["event_id"] for e in event_rows if int(e["end_index"]) < end]
    card = {
        "mode": "train_only_engineering_smoke",
        "region": "ny_bight",
        "station_ids": stations,
        "target": "water_level_m: observed total minus release deterministic FES2022 tide; metres",
        "context_hours": 72,
        "horizon_hours": 24,
        "stride_hours": 1,
        "release_contract": "96/72; this pilot deliberately uses 72/24",
        "train_rows": end,
        "train_row_interval": [0, end],
        "chronological_cut": cut,
        "embargo_hours": cut - end,
        "validation_windows": 0,
        "test_windows": 0,
        "unused_row_interval": [end, len(rows)],
        "train_windows": len(target),
        "unique_target_hours": len(residual) - 72,
        "independent_storm_count": None,
        "threshold_run_count": len(train_ids),
        "threshold_run_ids": train_ids,
        "mean": mean,
        "std": std,
        "normalization_fit_rows": end,
        "total_water_q95_q99": np.quantile(total, [0.95, 0.99]).tolist(),
        "thresholds_used_for_training": False,
        "future_covariates": False,
        "source_hashes": hashes,
        "license_status": "Owner authorized local engineering use; source license/redistribution "
        "and station datum sign-off remain incomplete per PR19; no redistribution",
        "generalization_evaluation_authorized": False,
    }
    for relative, actual_hash in hashes.items():
        if relative == "../MIGRATION_MANIFEST.json":
            continue
        entry = migration_entries[relative]
        if (
            actual_hash != entry["sha256"]
            or safe_file(root, relative).stat().st_size != entry["size_bytes"]
        ):
            raise ValueError(f"migration hash/size mismatch: {relative}")
    return history, target, card
