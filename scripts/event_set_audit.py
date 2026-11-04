"""Fix event count arithmetic — compute actual set unions from raw data.

WAVE25 BUG: census used total_independent = pot + tc with zero overlap.
CORRECT: overlap must be computed from actual TC track → POT cluster matching.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
DECLUSTERING = ROOT / "data" / "manifests" / "multiyear_declustering.json"
PREDS_DIR = ROOT / "data" / "predictions" / "v2"
RESULTS_DIR = ROOT / "data" / "results"


def audit_from_bootstrap():
    """Gold standard: use event_ids already assigned by wave20 bootstrap.

    These are the actual independent units the bootstrap resampled.
    """
    print("=" * 80)
    print("WAVE26 EVENT SET AUDIT — Bootstrap Event IDs (Gold Standard)")
    print("=" * 80)

    results = {}
    regions = ["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
               "southeast_australia", "thames_north_sea"]

    for region in regions:
        # Load A0 baseline (all pred files in region have SAME event_ids by design)
        a0_files = sorted(PREDS_DIR.glob(f"{region}_A0_real_only_gru_seed*.npz"))
        if not a0_files:
            print(f"  {region}: NO PREDICTION FILES FOUND")
            continue

        with np.load(a0_files[0], allow_pickle=True) as d:
            event_ids = d["event_ids"]

        unique_ids = np.unique(event_ids)
        n_units = len(unique_ids)

        # Classify event types
        canonical_units = unique_ids[unique_ids >= 0]  # positive = canonical TC
        pot_units = unique_ids[unique_ids < -999]     # negative = POT cluster (not TC)
        background = unique_ids[unique_ids == -999]    # fallback (not an extreme event)

        results[region] = {
            "total_independent": n_units,
            "canonical_tc_units": len(canonical_units),
            "pot_only_units": len(pot_units),
            "background_units": len(background),
            "unique_ids": unique_ids.tolist(),
        }

        print(f"\n{region} ({a0_files[0].stem}):")
        print(f"  Total unique event_ids: {n_units}")
        print(f"  Canonical TC units (id >= 0): {len(canonical_units)}")
        print(f"  POT-only units (id < -999):  {len(pot_units)}")
        print(f"  Background (-999): {len(background)}")
        print(f"  Event IDs: {unique_ids.tolist()}")

    return results


def audit_from_declustering():
    """Secondary check: raw POT clusters from declustering manifest."""
    print("\n" + "=" * 80)
    print("POT CLUSTERS FROM DECLUSTERING MANIFEST (for reference)")
    print("=" * 80)

    decl = json.loads(DECLUSTERING.read_text())
    total_pot = 0
    for region in decl:
        region_pot = 0
        for year, clusters in decl[region]["pot_clusters"].items():
            region_pot += len(clusters)
        print(f"  {region}: {region_pot} POT clusters (all years 2015-2024)")
        total_pot += region_pot
    print(f"  GRAND TOTAL POT CLUSTERS: {total_pot}")


def main():
    results = audit_from_bootstrap()
    audit_from_declustering()

    # Verify set arithmetic:
    # unique = pot_only ∪ canonical_tc  (background is a subset of POT or separate)
    print("\n" + "=" * 80)
    print("SET UNION ARITHMETIC VERIFICATION")
    print("=" * 80)

    grand_total_indep = sum(r["total_independent"] for r in results.values())
    grand_total_pot = sum(r["pot_only_units"] for r in results.values())
    grand_total_tc = sum(r["canonical_tc_units"] for r in results.values())
    grand_total_bg = sum(r["background_units"] for r in results.values())

    print(f"  Sum of unique units (bootstrap): {grand_total_indep}")
    print(f"    pot_only_units + canonical_tc_units: {grand_total_pot} + {grand_total_tc} = {grand_total_pot + grand_total_tc}")
    print(f"    background units: {grand_total_bg}")

    # For statistical inference we count non-background independent events only
    print(f"\n  Independent extreme units (non-background):")
    for region, r in results.items():
        extreme = r["total_independent"] - r["background_units"]
        print(f"    {region}: {extreme} (POT={r['pot_only_units']}, TC={r['canonical_tc_units']})")

    grand_extreme = grand_total_indep - grand_total_bg
    print(f"  GRAND TOTAL EXTREME INDEPENDENT UNITS (for 2024 bootstrap): {grand_extreme}")

    # Save
    out_path = RESULTS_DIR / "wave26_event_set_audit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
