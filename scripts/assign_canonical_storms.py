"""Wave 1.9 STORM EVENT INDEPENDENCE AUDIT + canonical assignment.

Scientific premise:
  TideTail's test split is entirely in 2024-01 to 2024-12. IBTrACS + HURDAT2
  catalog only TROPICAL CYCLONES (hurricanes, typhoons). 2024 coverage:
    us_gulf_coast:           6 tropical cyclones
    south_atlantic_charleston: 2 tropical cyclones
    ny_bight / aus / thames:  0 tropical cyclones in 2024

  Therefore, the primary independent-unit framework is OBSERVATIONAL EVENT
  SEGMENTATION using the 168-hour (7-day) support embargo per FORECAST_TASK_CONTRACT.
  Tropical cyclone matches (where they exist) provide bonus canonical_storm_ids
  for the same units. NEVER claim all units are canonical tropical cyclones.

  An independent event = a contiguous block of test windows where the gap
  between any two consecutive windows does NOT exceed 168 hours. Any gap
  >= 168h starts a NEW independent event. This is a well-established
  declustering rule for water-level extremes (Coles 2001 §5.2.3).

Process per region:
  1. Load IBTrACS+HURDAT2, dedup, filter regional storms 2024
  2. 168h segmentation of test timestamps → observ_seg_ids
  3. Assign canonical_storm_id to observ segments where storm track matches
  4. Independent unit = (canonical_storm_id if matched, else -(observ_seg_id+1))
  5. Group storm-level bootstrap at independent-unit level
  6. Write updated prediction .npz files

Outputs:
  - Updated prediction .npz files (all 190)
  - data/results/wave19_storm_bootstrap.json
  - docs/audits/STORM_EVENT_INDEPENDENCE_AUDIT.md (written separately)
"""
from __future__ import annotations

import json, os, sys, math, shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tidetail.storm_events import (
    load_ibtracs_csv, load_hurdat2,
    parse_ibtracs_to_storms, parse_hurdat2_to_storms,
    dedup_ibtracs_hurdat2, storms_in_region,
)

DATA_ROOT = Path("/data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast")
PREDS_DIR = ROOT / "data" / "predictions" / "v2"

REGION_BBOX = {
    "ny_bight":                  {"lat": (38.0, 42.0),  "lon": (-76.0, -72.0)},
    "us_gulf_coast":             {"lat": (26.0, 31.0),  "lon": (-97.0, -84.0)},
    "south_atlantic_charleston": {"lat": (31.0, 34.5),  "lon": (-81.0, -78.0)},
    "southeast_australia":      {"lat": (-38.0, -30.0),"lon": (148.0, 154.0)},
    "thames_north_sea":          {"lat": (50.0, 53.5),  "lon": (-2.0, 2.0)},
}

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_global_storms():
    RAW = Path("/data/yingda/tidetail-data/raw")
    REAL = Path("/data/yingda/tidetail-data/real-sources-v1")
    ib = parse_ibtracs_to_storms(load_ibtracs_csv(RAW / "ibtracs/v04r01/ibtracs.since1980.list.v04r01.csv"))
    h2 = parse_hurdat2_to_storms(load_hurdat2(REAL / "hurdat2/hurdat2-1851-2025.txt"))
    all_s = dedup_ibtracs_hurdat2(ib, h2)
    print(f"  Global deduped: {len(all_s)} storms (IBTrACS={len(ib)}, HURDAT2={len(h2)})")
    return all_s

def observ_segment_ids(timestamps_ns, gap_hours=168):
    """168h-embargo observational event segmentation.

    Consecutive test windows separated by <168h → SAME observ segment.
    Gap >=168h starts a NEW independent segment. This ensures no two
    forecast windows from different segments share unmodeled residual
    correlation. Per FORECAST_TASK_CONTRACT.md §LEAKAGE_PREVENTION.
    """
    ts = timestamps_ns.astype(np.int64)
    N = len(ts)
    seg = np.full(N, -1, dtype=np.int32)
    cur = 0
    seg[0] = 0
    for i in range(1, N):
        if (ts[i] - ts[i-1]) / 1e9 / 3600.0 > gap_hours:
            cur += 1
        seg[i] = cur
    return seg

def storm_level_bootstrap(tw_pred, tw_obs, group_ids, n_boot=500, seed=42):
    """Bootstrap resampling at INDEPENDENT UNIT level.

    Each unit (canonical storm OR observ segment) is one draw. All
    windows within a unit stay together. This is correct because
    residual water levels within a 72h forecast window are highly
    correlated (intra-group dependence), but across units they should
    be approximately independent given the 168h gap embargo.
    """
    rng = np.random.default_rng(seed)
    errors = np.abs(tw_pred - tw_obs)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    group_means = np.array([np.mean(errors[group_ids == g]) for g in unique_groups])

    boot = np.zeros(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        boot[b] = np.mean(group_means[chosen])

    return {
        "independent_units": int(n_groups),
        "point_mae": float(np.mean(group_means)),
        "bootstrap_mean": float(np.mean(boot)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_boot": n_boot,
    }

def process_region(region, storms_2024):
    print(f"\n{'='*60}\n{region}\n{'='*60}")
    base = DATA_ROOT / region
    wi = np.load(base / "window_index_v1.npz")
    all_ts = wi["all_timestamps_ns"]
    test_starts = wi["test_start_indices"]
    test_ts_ns = all_ts[test_starts]
    N = len(test_ts_ns)
    test_dates = [datetime.fromtimestamp(ts / 1e9, tz=None) for ts in test_ts_ns]
    print(f"  Test windows: {N}, range: {min(test_dates).date()} – {max(test_dates).date()}")

    # Step 1: 168h observ segmentation (the core independent-unit framework)
    seg_ids = observ_segment_ids(test_ts_ns, gap_hours=168)
    n_segments = len(np.unique(seg_ids))
    print(f"  168h-embargo observ segments: {n_segments}")

    # Step 2: Overlay canonical tropical cyclone storms (where applicable)
    bbox = REGION_BBOX[region]
    tc_2024 = storms_in_region(
        storms_2024,
        lat_range=(bbox["lat"][0], bbox["lat"][1]),
        lng_range=(bbox["lon"][0], bbox["lon"][1]),
    )
    print(f"  IBTrACS/HURDAT2 tropical cyclones 2024 in bbox: {len(tc_2024)}")

    canonical_ids = np.full(N, -1, dtype=np.int32)
    storm_source = np.full(N, "", dtype=object)
    storm_valid = np.zeros(N, dtype=np.bool_)
    storm_id_map = {}

    for i, d in enumerate(test_dates):
        for storm in tc_2024:
            lead = storm.event_start_utc.replace(tzinfo=None) - timedelta(hours=24) if storm.event_start_utc.tzinfo else storm.event_start_utc - timedelta(hours=24)
            lag = storm.event_end_utc.replace(tzinfo=None) + timedelta(hours=72) if storm.event_end_utc.tzinfo else storm.event_end_utc + timedelta(hours=72)
            if lead <= d <= lag:
                if storm.canonical_storm_id not in storm_id_map:
                    storm_id_map[storm.canonical_storm_id] = len(storm_id_map)
                canonical_ids[i] = storm_id_map[storm.canonical_storm_id]
                storm_source[i] = storm.source_dataset
                storm_valid[i] = True
                break

    n_storm_matched = int((canonical_ids != -1).sum())
    print(f"  Storm-matched windows: {n_storm_matched}/{N} ({n_storm_matched/N*100:.1f}%)")

    # Step 3: Combined independent unit
    # Pos = canonical storm index, Neg = -(segment_id + 1)
    combined_units = np.where(canonical_ids != -1, canonical_ids, -(seg_ids + 1))
    n_units = len(np.unique(combined_units))
    n_seg_only = n_units - len(storm_id_map)
    print(f"  Independent units: {n_units} ({len(storm_id_map)} canonical TCs + {n_seg_only} observ segments)")

    # Step 4: Update all 190 prediction files + compute storm-level CIs
    region_files = sorted(PREDS_DIR.glob(f"{region}_*_seed*.npz"))
    # Build backup dir once
    backup_dir = PREDS_DIR / "v2_before_storm_ids"
    backup_dir.mkdir(exist_ok=True)

    results = {}
    for pf in region_files:
        if not (backup_dir / pf.name).exists():
            shutil.copy2(pf, backup_dir / pf.name)

        with np.load(pf, allow_pickle=True) as d:
            data_dict = dict(d)
            tw_pred = data_dict["pred_residual"] + data_dict["future_tide"]
            tw_obs = data_dict["obs_residual"] + data_dict["future_tide"]
            meta = json.loads(data_dict["meta_json"].item()) if "meta_json" in data_dict else {}

        bs = storm_level_bootstrap(tw_pred, tw_obs, combined_units)

        # Update event_ids to independent-unit IDs
        data_dict["event_ids"] = combined_units
        data_dict["canonical_storm_ids"] = canonical_ids
        data_dict["storm_valid"] = storm_valid.astype(np.int32)
        data_dict["storm_source"] = storm_source.reshape(-1, 1)  # object array compatible
        data_dict["observ_segment_ids"] = seg_ids
        data_dict["meta_json"] = np.array([json.dumps({
            **meta,
            "independent_units": n_units,
            "canonical_tc_count": len(storm_id_map),
            "observ_segment_count": n_segments,
            "storm_matched_frac": n_storm_matched / N,
            "bootstrap": bs,
            "backfill_version": "wave19_storm_integration_v1",
            "declustering": "168h gap embargo (Coles 2001)",
        })], dtype=object)

        np.savez(pf, **data_dict)

        cfg = meta.get("group", pf.stem.split("_")[1])
        model = meta.get("model", pf.stem.split("_")[2] if len(pf.stem.split("_")) > 2 else "?")
        seed = meta.get("seed", 42)
        key = f"{cfg} {model} seed{seed}"
        results[key] = bs

        if cfg == "A0_real_only" and model == "gru" and seed == 42:
            print(f"    A0 GRU seed42: MAE={bs['point_mae']:.4f}, CI=[{bs['ci_low']:.4f},{bs['ci_high']:.4f}], units={bs['independent_units']}")

    return {
        "region": region,
        "test_windows": N,
        "test_date_range": [str(min(test_dates).date()), str(max(test_dates).date())],
        "independent_units": n_units,
        "canonical_tcs": len(storm_id_map),
        "observ_segments": n_segments,
        "storm_matched_frac": n_storm_matched / N,
        "canonical_ids_map": {v: k for k, v in storm_id_map.items()},
        "bootstrap_results": results,
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Wave 1.9 STORM EVENT INDEPENDENCE AUDIT")
    print("Independent unit = 168h-embargo observ segment (+ canonical TC overlay)")
    print("=" * 70)

    all_storms = load_global_storms()
    results = []
    for r in ["ny_bight", "us_gulf_coast", "south_atlantic_charleston",
              "southeast_australia", "thames_north_sea"]:
        results.append(process_region(r, all_storms))

    # Save
    out = ROOT / "data" / "results" / "wave19_storm_bootstrap.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))

    # Print verdicts
    print(f"\n{'='*80}")
    print("STORM-LEVEL BOOTSTRAP CI VERDICTS (n_boot=500, n_independent_units per region)")
    print(f"{'='*80}")
    for r in results:
        print(f"\n{r['region']}: {r['independent_units']} independent units "
              f"({r['canonical_tcs']} canonical TCs + {r['observ_segments']} observ segments)")
        a0_key = "A0_real_only gru seed42"
        a0 = r["bootstrap_results"].get(a0_key)
        if not a0:
            print("  NO A0 baseline found")
            continue
        print(f"  A0 GRU seed42: MAE={a0['point_mae']:.4f}, CI=[{a0['ci_low']:.4f},{a0['ci_high']:.4f}]")
        # Per-A3 verdict: better if candidate CI_high < A0 CI_low
        verdicts = {"better": 0, "worse": 0, "overlap": 0}
        for cfg, bs in r["bootstrap_results"].items():
            if cfg.startswith("A3") and "gru" in cfg:
                if bs["ci_high"] < a0["ci_low"]:
                    verdicts["better"] += 1
                    v = "✅ BETTER"
                elif bs["ci_low"] > a0["ci_high"]:
                    verdicts["worse"] += 1
                    v = "❌ WORSE"
                else:
                    verdicts["overlap"] += 1
                    v = "➖ OVERLAP"
                delta = (bs["point_mae"] - a0["point_mae"]) / a0["point_mae"] * 100
                print(f"  {v} {cfg}: Δ%={delta:+.2f}, MAE={bs['point_mae']:.4f}, CI=[{bs['ci_low']:.4f},{bs['ci_high']:.4f}]")
        print(f"  Totals: better={verdicts['better']}, worse={verdicts['worse']}, overlap={verdicts['overlap']}")

    print(f"\nSummary → {out}")
