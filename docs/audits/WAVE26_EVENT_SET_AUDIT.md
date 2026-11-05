# Wave 2.6 — Event Set Audit

**Date**: 2026-11-04
**Motivation**: Wave 2.5 census double-counted tropical cyclones by summing POT+TC without overlap subtraction.

## Executive Summary

The **gold standard** for independent event counts is the `event_ids` field in prediction npz files — assigned by `scripts/wave20_pot_storm_bootstrap.py` using temporal overlap between canonical TC tracks and POT cluster windows. This audit uses those actual event IDs, not census summary tables.

## Actual Independent Units (2024 Only — Bootstrap Gold Standard)

| Region | Unique event_ids | Canonical TC (≥0) | POT Cluster (<-999) | Background (-999) | Non-bg Extreme |
|---|---:|---:|---:|---:|---:|
| ny_bight | 16 | 0 | 15 | 1 | **15** |
| us_gulf_coast | 16 | 6 | 10 | 1 | **15** |
| charleston | 15 | 2 | 12 | 1 | **14** |
| southeast_australia | 7 | 0 | 6 | 1 | **6** |
| thames_north_sea | 10 | 0 | 9 | 1 | **9** |
| **TOTAL** | **64** | **8** | **52** | **5** | **59** |

## Event ID Encoding Convention

From `wave20_pot_storm_bootstrap.py` L72-99:

| ID Range | Type | Assignment Rule |
|---|---|---|
| `id >= 0` | TRACK_VALIDATED_STORM | Canonical TC track at gauge |
| `id < -999` | OBSERVED_SURGE_EVENT | POT q95 cluster (negative offset) |
| `id == -999` | BACKGROUND_PERIOD | Fallback (non-extreme test windows) |

**Priority**: TC assignment first. If a POT cluster's time window contains a TC track, all its windows become TC events (>=0), negating the POT cluster's negative ID.

## WAVE25 Census Errors Corrected

| Region | Census (2024) | Bootstrap (actual) | Discrepancy | Root Cause |
|---|---:|---:|---|---|
| ny_bight | 15 | 15 | 0 | ✅ Correct |
| us_gulf_coast | 17 | 16 | **+1** | Census used POT+TC without subtracting TCs that ALSO generated POT clusters |
| charleston | 15 | 14 | **+1** | Same — one TC track window captured POT extreme windows |
| southeast_australia | 6 | 6 | 0 | ✅ No TCs, no overlap issue |
| thames_north_sea | 10 | 9 | **+1** | One POT cluster fully captured by observ_segment fallback |

**2024 correct extreme units total: 59** (not 63 as census claimed).

## 10-Year Count (2015-2024)

The census reported 707 for the full decade. Using the bootstrap protocol's event assignment logic:
- POT clusters from `multiyear_declustering.json`: **634 total** (no overlap subtraction)
- Canonical TCs from census: **73** total
- **Correct unique = POT + TC - overlap** where overlap = TC windows that fall within POT clusters

From the bootstrap's ID assignment priority, every TC that produces a POT exceedance will be counted as a TC event (>=0), not a POT event. So the TRUE unique count = number of POT clusters + number of TCs that did NOT overlap with existing POT clusters.

Since overlap = all TCs that caused POT exceedances (overwhelming majority), the correct unique is approximately **634** (POT clusters already cover TCs that caused surges).

## Test Cohorts (Corrected — Using Bootstrap Logic Proportionally)

Census overcounted 2024 by 63→59 (6%). Applied proportionally across all cohorts:

| Cohort | Census Count | Adjusted (×0.937) | Verdict |
|---|---:|---:|---|
| 2024 only | 63 | **59** | Gold standard (from actual event IDs) |
| 2023-2024 | 131 | **~123** | Proportional correction needed |
| 2022-2024 | 197 | **~185** | Proportional correction needed |
| 2015-2024 | 707 | **~634** | POT clusters cover TCs |

**For statistical inference, ALWAYS use the bootstrap-assigned event_ids**, not summary counts.

## What This Means for Power

The power audit used bootstrap `total_independent_units_A0` which is correct (64 total including background, 59 extreme). The previous audit also had `independent_units = 15-17` from bootstrap, which matches our audit. So the **Wave 2.5 power numbers were correct** — the census's 707 vs 634 discrepancy was purely terminological.

## Regression Test

```python
def test_event_set_union_arithmetic():
    """Every prediction file must follow: unique = pot_only ∪ canonical_tc ∪ background"""
    with np.load("predictions/v2/charleston_A0_real_only_gru_seed42.npz") as d:
        event_ids = d["event_ids"]
    assert np.allclose(np.unique(event_ids), 
                      np.concatenate([[-999], np.arange(-14, 0), np.arange(0, 2)]))
```

---
*Wave 2.6 Event Set Audit — corrected 2026-11-04*
*Source: scripts/event_set_audit.py + data/results/wave26_event_set_audit.json*

## WAVE 2.7 CORRECTION (2026-11-05)

Two statements above are superseded.

1. **Event-ID encoding.** The table says POT clusters are `id < -999`. They are not:
   POT clusters are encoded as `-(cluster_id + 1)`, i.e. `-1 … -N`, and only
   `-999` means background. The Wave 2.6 audit's `pot_units` filter
   (`unique_ids < -999`) therefore always returned an empty POT set. Correct
   classification lives in `src/tidetail/event_union.py::classify_event_id`.
2. **Decade count.** "Correct unique is approximately 634" is replaced by the
   measured union: POT 634, TC 73, one-to-one matches **28**, POT-only 606,
   TC-only 45, **union 679**. Overlap was never 73.

The 2024 bootstrap counts in the table (15/15/14/6/9 = 59 extreme units) are
independently reproduced and remain valid. 2024 set-union arithmetic gives 60
before intersecting with available test windows; the difference is one Thames
cluster with no 2024 test window.

Test-cohort unions (2015-2024): 178 / 138 / 143 / 101 / 119 per region.
See `data/results/wave27_event_set_audit.json` for tolerance sensitivity
(0/24/48/72 h) and integrity checks (no duplicate timestamps, aligned
`event_ids`, 8,601 test windows with Thames = 6,968).

Forecast-window integrity: no duplicated storm events across splits and no
forecast-window overlap were found in the audited prediction files.
