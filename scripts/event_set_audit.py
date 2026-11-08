"""Wave 2.7 independent event-set audit.

Recomputes the POT/TC union from raw manifests and verifies it against the
wave20 bootstrap ``event_ids`` actually used to group forecast windows.

Wave 2.5 claimed POT=634, TC=73, overlap=73, total=707. That is arithmetically
impossible. This script computes an explicit one-to-one match and reports the
identity ``union = POT + TC - matched``.

Usage:
    python3 scripts/event_set_audit.py

Outputs:
    data/results/wave26_event_set_audit.json   (corrected Wave 2.6 artifact)
    data/results/wave27_event_set_audit.json  (Wave 2.7 verification)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tidetail.event_union import classify_event_id, match_one_to_one, union_cardinality

DECLUSTERING = ROOT / "data" / "manifests" / "multiyear_declustering.json"
PREDS_DIR = ROOT / "data" / "predictions" / "v2"
RESULTS_DIR = ROOT / "data" / "results"

REGIONS = [
    "ny_bight",
    "us_gulf_coast",
    "south_atlantic_charleston",
    "southeast_australia",
    "thames_north_sea",
]
PERIODS = {
    "2024": [2024],
    "2023-2024": [2023, 2024],
    "2022-2024": [2022, 2023, 2024],
    "2015-2024": list(range(2015, 2025)),
}
TOLERANCES_H = [0.0, 24.0, 48.0, 72.0]


def period_inputs(manifest: dict, region: str, years: list[int]) -> tuple[list[dict], list[str]]:
    pot: list[dict] = []
    tc_dates: list[str] = []
    for year in years:
        pot.extend(manifest[region]["pot_clusters"].get(str(year), []))
        for entry in manifest[region]["tcs"].get(str(year), []):
            # tcs entries are [canonical_storm_id, peak_wind_kt, date]
            tc_dates.append(entry[2])
    return pot, tc_dates


def audit_manifest(manifest: dict) -> dict:
    out = {}
    for period_name, years in PERIODS.items():
        out[period_name] = {"regions": {}, "grand": {}}
        grand_pot = grand_tc = grand_matched = 0
        for region in REGIONS:
            pot, tc_dates = period_inputs(manifest, region, years)
            primary = union_cardinality(
                match_one_to_one(tc_dates, pot, tolerance_hours=0.0)
            )
            sensitivity = {
                f"tol_{int(tol)}h": union_cardinality(
                    match_one_to_one(tc_dates, pot, tolerance_hours=tol)
                )
                for tol in TOLERANCES_H
            }
            out[period_name]["regions"][region] = {
                "primary_strict": primary,
                "tolerance_sensitivity": sensitivity,
            }
            grand_pot += primary["pot_total"]
            grand_tc += primary["tc_total"]
            grand_matched += primary["matched_overlap"]

        grand = union_cardinality(
            {
                "matched": [(0, 0)] * grand_matched,
                "pot_only": [0] * (grand_pot - grand_matched),
                "tc_only": [0] * (grand_tc - grand_matched),
                "pot_total": grand_pot,
                "tc_total": grand_tc,
            }
        )
        out[period_name]["grand"] = grand
    return out


def _event_ids_for_region(region: str) -> np.ndarray:
    files = sorted(PREDS_DIR.glob(f"{region}_A0_real_only_gru_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no A0 prediction file for {region}")
    with np.load(files[0], allow_pickle=True) as d:
        return d["event_ids"].astype(np.int64)


def audit_bootstrap_windows() -> dict:
    """Verify window-level grouping and 2024 independent-unit counts."""
    out = {}
    for region in REGIONS:
        ids = _event_ids_for_region(region)
        unique = np.unique(ids)
        kinds = [classify_event_id(int(u)) for u in unique]
        out[region] = {
            "total_units": int(len(unique)),
            "tc_units": int(sum(k == "tc" for k in kinds)),
            "pot_units": int(sum(k == "pot" for k in kinds)),
            "background_units": int(sum(k == "background" for k in kinds)),
            "extreme_units": int(sum(k != "background" for k in kinds)),
            "total_windows": int(len(ids)),
            "background_windows": int(np.sum(ids == -999)),
            "unit_ids": [int(u) for u in unique],
        }
    out["grand"] = {
        "extreme_units": int(sum(v["extreme_units"] for v in out.values())),
        "tc_units": int(sum(v["tc_units"] for v in out.values())),
        "pot_units": int(sum(v["pot_units"] for v in out.values())),
        "total_windows": int(sum(v["total_windows"] for v in out.values())),
    }
    return out


def verify_window_integrity() -> dict:
    """Check that each region's windows are unique and event groups are disjoint."""
    checks = {}
    for region in REGIONS:
        ids = _event_ids_for_region(region)
        files = sorted(PREDS_DIR.glob(f"{region}_A0_real_only_gru_seed*.npz"))
        with np.load(files[0], allow_pickle=True) as d:
            ts = d["timestamps"].astype(np.int64)
        dup = int(len(ts) - len(np.unique(ts)))
        checks[region] = {
            "n_windows": int(len(ts)),
            "duplicate_timestamps": dup,
            "unit_count": int(len(np.unique(ids))),
            "event_ids_len_matches_windows": bool(len(ids) == len(ts)),
        }
    return checks


def main() -> None:
    manifest = json.loads(DECLUSTERING.read_text())

    print("=" * 78)
    print("WAVE 2.7 EVENT SET AUDIT — explicit one-to-one union")
    print("=" * 78)

    manifest_audit = audit_manifest(manifest)
    for period, payload in manifest_audit.items():
        g = payload["grand"]
        print(
            f"  {period:>9}: POT={g['pot_total']:3d} TC={g['tc_total']:3d} "
            f"matched={g['matched_overlap']:3d} POT_only={g['pot_only']:3d} "
            f"TC_only={g['tc_only']:3d} UNION={g['union']:3d}"
        )
        assert g["union"] == g["pot_total"] + g["tc_total"] - g["matched_overlap"]

    print("\n  Tolerance sensitivity for 2015-2024 (matched overlap):")
    for tol_key, val in manifest_audit["2015-2024"]["regions"][
        "us_gulf_coast"
    ]["tolerance_sensitivity"].items():
        print(f"    us_gulf_coast {tol_key}: matched={val['matched_overlap']}, union={val['union']}")

    boot = audit_bootstrap_windows()
    print("\n  Bootstrap window grouping (2024 test year):")
    for region in REGIONS:
        v = boot[region]
        print(
            f"    {region:>26}: units={v['total_units']:2d} "
            f"(extremes={v['extreme_units']:2d}, TC={v['tc_units']}, POT={v['pot_units']}, "
            f"bg={v['background_units']}); windows={v['total_windows']}, "
            f"bg_windows={v['background_windows']}"
        )
    print(f"    GRAND extremes={boot['grand']['extreme_units']}, "
          f"TC units={boot['grand']['tc_units']}, POT units={boot['grand']['pot_units']}")

    integrity = verify_window_integrity()
    print("\n  Window integrity:")
    for region, v in integrity.items():
        print(f"    {region:>26}: windows={v['n_windows']}, "
              f"duplicate_ts={v['duplicate_timestamps']}, "
              f"event_ids_aligned={v['event_ids_len_matches_windows']}")

    # The 2024 manifest union should equal the bootstrap extreme-unit count
    # for every region where every cluster has test windows. Thames loses one
    # cluster with no test window, so compare per region with an explicit note.
    print("\n  2024 cross-check (manifest union vs bootstrap extreme units):")
    cross = {}
    for region in REGIONS:
        mu = manifest_audit["2024"]["regions"][region]["primary_strict"]["union"]
        bu = boot[region]["extreme_units"]
        cross[region] = {"manifest_union": mu, "bootstrap_extreme_units": bu,
                         "delta": bu - mu}
        print(f"    {region:>26}: manifest={mu:2d} bootstrap={bu:2d} delta={bu - mu:+d}")

    payload = {
        "audit_version": "wave27_v1",
        "identity_checked": "union = POT + TC - matched_overlap",
        "periods": manifest_audit,
        "bootstrap_2024": boot,
        "window_integrity": integrity,
        "cross_check_2024": cross,
        "corrections": {
            "wave25_claim": "POT=634, TC=73, overlap=73, total=707",
            "wave25_verdict": "INVALID — 634+73-73=634, not 707; overlap was never computed",
            "wave27_2015_2024": manifest_audit["2015-2024"]["grand"],
            "wave27_2024_bootstrap_extremes": boot["grand"]["extreme_units"],
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    for name in ("wave26_event_set_audit.json", "wave27_event_set_audit.json"):
        path = RESULTS_DIR / name
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
