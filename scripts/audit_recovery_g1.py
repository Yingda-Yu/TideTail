"""Read-only follow-up on frozen G1 blockers; write aggregates, never source rows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tidetail.pilot_data import OWNER_ROOT, digest, safe_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(safe_file(OWNER_ROOT, "file_inventory_sha256.json").read_text())
    release = {e["path"]: e for e in inventory["files"]}
    migration_path = safe_file(OWNER_ROOT.parent, "MIGRATION_MANIFEST.json")
    migration = json.loads(migration_path.read_text())
    assert migration["destination"] == str(OWNER_ROOT)
    entries = {e["path"]: e for e in migration["file_entries"]}
    hashes = {"../MIGRATION_MANIFEST.json": digest(migration_path)}

    def checked(relative):
        path = safe_file(OWNER_ROOT, relative)
        actual = digest(path)
        assert actual == entries[relative]["sha256"]
        assert path.stat().st_size == entries[relative]["size_bytes"]
        if relative in release:
            assert actual == release[relative]["sha256"]
            assert path.stat().st_size == release[relative]["bytes"]
        hashes[relative] = actual
        return path

    def rows(relative):
        with checked(relative).open() as handle:
            return list(csv.DictReader(handle))

    availability = json.loads(checked("region_availability.json").read_text())
    result = {"migration_present_and_checked": True, "regions": {}}
    for region in availability["regions"]:
        name = region["region"]
        prefix = f"derived/prime_coast/{name}/"
        meta = json.loads(checked(prefix + "train_only_metadata_v1.json").read_text())
        events = rows(prefix + "event_table_v1.csv")
        cuts = [meta["train_end_index"], meta["validation_end_index"]]
        crossing = [{"event_id": e["event_id"], "start": int(e["start_index"]),
                     "end": int(e["end_index"]), "cut": cut, "assigned_split": e["split"]}
                    for e in events for cut in cuts
                    if int(e["start_index"]) < cut <= int(e["end_index"])]
        entry = {"formal_training_authorized": region["formal_training_authorized"],
                 "crossing_events": crossing, "cut_indices": cuts,
                 "independent_storm_count": None}
        if name in ("pearl_river_hong_kong", "thames_north_sea"):
            train = rows(prefix + "target_timeline_v1.csv")[:cuts[0]]
            forcing = rows(prefix + "forcing_timeline_v1.csv")[:cuts[0]]
            tide = rows(prefix + "tide_timeline_v1.csv")[:cuts[0]]
            assert [r["timestamp_utc"] for r in train] == [r["timestamp_utc"] for r in forcing]
            assert [r["timestamp_utc"] for r in train] == [r["timestamp_utc"] for r in tide]
            valid = np.array([r["is_missing"] == "0" and np.isfinite(float(r["water_level_m"]))
                              for r in train])
            entry["target_valid_count"] = int(valid.sum())
            entry["scaler_probes"] = {}
            for column, scaler in meta["scalers"].items():
                source = train if column == "water_level_m" else (
                    tide if column == "tide_level_m" else forcing)
                values = np.array([float(r[column]) if r[column] else np.nan for r in source])
                common = valid & np.isfinite(values)
                entry["scaler_probes"][column] = {
                    "stored_count": scaler["count"], "per_column_finite_count":
                    int(np.isfinite(values).sum()), "target_valid_finite_count": int(common.sum()),
                    "common_mask_mean_delta": float(values[common].mean()-scaler["mean"]),
                    "common_mask_std_delta": float(values[common].std()-scaler["std"])}
        result["regions"][name] = entry
    result["source_hashes"] = hashes
    result["claim"] = ("Common-target-mask numerical reproduction only; builder provenance, "
                       "license/datum and independent-storm G1 signoff remain unresolved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"regions": len(result["regions"]), "crossing_events":
                      {k: v["crossing_events"] for k, v in result["regions"].items()}}))


if __name__ == "__main__":
    main()
