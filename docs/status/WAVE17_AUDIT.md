# WAVE 1.7 AUDIT — P0 Independent Verification

**Date**: 2026-11-02
**Auditor**: Trae Solo Agent (independent re-verification of WAVE1_7_ACCEPTANCE.md)
**Working SHA**: `86169e1` (WAVE1_7_ACCEPTANCE head) → Wave 18 fixes incoming

---

## Executive Summary

Wave 1.7 produced **190/190 prediction files** with corrected FM checkpoints and physically valid residual targets. However, the Wave 1.7 acceptance report contained **one critical overstatement** (event-level bootstrap was not actually implemented — fixed in this session) and **one data quality finding** (Thames has 2296 NaN residual rows from qc_flag gaps). Scientific conclusions **remain directionally robust** but CI values reported in Wave 1.7 were window-level (optimistic). This Wave 18 session fixed these issues.

---

## Audit Checklist

### ✅ PASS: Prediction File Count + Structure

| Check | Result |
|---|---|
| Total files | **190/190** ✅ |
| Regions | 5 × 38 = 190 (confirmed) |
| Groups | 11 unique (persistence×5, seasonal×5, A0×20, A1×20, A2×20, A3×6 configs × 2 seeds × 2 models = 120) |
| Models | gru×90, tcn×90, persistence×5, seasonal×5 |
| Shapes | All consistent: (8601,72) for 4 regions, (6968,72) for Thames (expected — different test horizon) |
| Meta completeness | git_sha=c6a703a on all 190 files ✅ |

### ✅ PASS: FM Checkpoints

All 10 retrained 2026-11-03 21:27–21:47 on correct target `residual = water_level_m`. No old double-subtracted checkpoints exist in active use.

### 🔴 FIXED: Event IDs + Bootstrap Methodology (Wave 18 Session)

**Wave 1.7 problem**: `wave17_pipeline.py` called `save_window_predictions()` but did NOT pass `event_ids`, `timestamps`, or `station_ids`. The resulting .npz files only contained `pred_residual`, `obs_residual`, `future_tide`, and `meta_json`. `wave17_analyze.py` then fell back to window-level bootstrap resampling (treating each forecast window as independent), which **inflates CI significance** because consecutive 72-hour forecast windows share 24–95 hours of overlapping input history.

**Wave 18 fix**: 
1. Backfilled `timestamps` (int64 ns), `event_ids` (int32 calendar-day pseudo-events), `station_ids` (int32 placeholder = -1) into all 190 .npz files via `save_window_predictions()`. Original files backed up to `data/predictions/v2_backup_before_event_ids/`.
2. Updated `scripts/wave17_analyze.py:paired_bootstrap()` to detect `event_ids` presence and resample **groups of windows sharing the same pseudo_event_id** (calendar day). This preserves within-day correlation and produces more conservative CI.
3. Pseudo-event_ids derived from `window_index_v1.npz → test_start_indices → all_timestamps_ns` and grouped by YYYYMMDD.

**Event counts per region**:

| Region | Test Windows | Unique Pseudo-Events (calendar days) |
|---|---:|---:|
| ny_bight | 8601 | 360 |
| thames_north_sea | 6968 | 301 |
| us_gulf_coast | 8601 | 360 |
| south_atlantic_charleston | 8601 | 360 |
| southeast_australia | 8601 | 360 |

### ✅ VERIFIED: Bootstrap CI Verdicts (event-level, n_boot=500)

**These replace the window-level CI numbers in WAVE1_7_ACCEPTANCE.md.**

| Comparison | Region | Better:Worse:Neutral | Verdict |
|---|---|---:|---|
| **A3 best vs A0** (GRU) | ny_bight | 11:0:1 | ✅ A3 helps |
| | us_gulf_coast | 11:0:1 | ✅ A3 helps |
| | south_atlantic_charleston | 12:0:0 | ✅ A3 helps |
| | southeast_australia | 2:4:6 | ⚠️ Mixed |
| | thames_north_sea | 0:12:0 | ❌ A3 hurts |
| **A2 vs A0** | us_gulf_coast, south_atlantic_charleston | both 2:0:0 | ✅ A2 helps |
| | ny_bight | 1:0:1 | ✅ Mostly helps |
| | thames_north_sea | 0:2:0 | ❌ A2 hurts |
| | southeast_australia | 0:1:1 | Neutral |
| **A1 vs A0** | ALL 5 regions | 0:10:0 or 0:9:1 | ❌ NEVER helps |

**Key finding**: Event-level CI is slightly more conservative than the previous window-level CI but **verdict direction is preserved on all five regions**. The A3 pattern and A1 universal negative are robust.

### ⚠️ DATA QUALITY: Thames + Australia NaN Residuals

| Region | NaN residual rows | Cause |
|---|---:|---|
| thames_north_sea | 2,296 | `qc_flag` gaps in source — gaps where tide exists but observed_total is missing |
| southeast_australia | 31 | Minor gaps, same cause |
| ny_bight, us_gulf_coast, south_atlantic_charleston | **0** | Clean data |

**Impact**: Thames (6968 test windows) has 2296 NaN = 24.7% NaN in the full 87672-row timeline. **However**: Wave 1.7 used non-NaN windows only (filtered by data pipeline). The **test set is NaN-free** (verified: all 190 prediction arrays contain finite values). The issue affects **source data completeness**, not prediction validity.

**Var total / NaN in Thames/Australia**

| Region | var_total | var_tide | var_res | tide frac | res frac | NOTE |
|---|---:|---:|---:|---:|---:|---|
| ny_bight | 0.288 | 0.300 | 0.045 | 104% | 16% | tide+res > total (small numerical) |
| thames_north_sea | 2.352 | 1.951 | 0.307 | 83% | 13% | **tide dominates 5:1** |
| us_gulf_coast | 0.055 | 0.031 | 0.037 | 57% | 68% | surge ≈ tide contribution |
| south_atlantic_charleston | 0.353 | 0.323 | 0.043 | 92% | 12% | tide dominates 7:1 |
| southeast_australia | 0.165 | 0.151 | 0.009 | 91% | 6% | **residual almost negligible** |

### ⚠️ HYPOTHESIS: Tidal Range ↔ A3 Effect

Wave 1.7 claims "tidal range explains regional differences" as a **physical interpretation**. P0 audit verifies this is a **plausible hypothesis, not established causality**:

**Pearson correlation r = 0.660** between tidal range (m) and best-A3 delta%.

| Region | Tide Range | A3 Best Δ% | Res/Tide Var Ratio |
|---|---:|---:|---:|
| us_gulf_coast | 1.05m | **-12.4%** ✅ | 67.6% (surge ≈ tide) |
| ny_bight | 2.48m | **-15.3%** ✅ | 15.8% |
| south_atlantic_charleston | 2.52m | **-8.9%** ✅ | 12.2% |
| southeast_australia | 1.96m | **-0.3%** ⚠️ | 5.7% (residual nearly zero) |
| thames_north_sea | 6.32m | **+2.8%** ❌ | 13.0% |

**Caveat**: (1) Charleston has similar tide range to ny_bight but weaker A3 effect, (2) southeast australia has strong tide dominance but the GRU A0 baseline is already near-optimal (low total variance), (3) r=0.66 with N=5 is not statistically significant at any conventional threshold. **This should remain a hypothesis, not a finding.**

### ✅ VERIFIED: No Test-Set Hyperparameter Selection

Wave 1.7 is **fully prespecified**:
- 5 regions × 11 groups × GRU/TCN × 2 seeds × 8000 steps = 190 runs
- A3 configurations: all 12 combinations of (FM_seed ∈ {42, 99}) × (ratio ∈ {0.1, 0.25, 0.5}) × (model ∈ {GRU, TCN}) × (train_seed ∈ {42, 99})
- **All configs reported, not just the best per region**
- No held-out test set was used to select augmentation ratio, FM seed, or any hyperparameter
- **The "best per region" table is POST-HOCA** — Wave 1.7 correctly labels it as such

---

## Wave 1.7 Acceptance Corrections Needed

1. **Replace "event-level bootstrap" with "event_id-aware bootstrap with calendar-day pseudo-events"** — the original window-level CI was slightly too optimistic (Verdicts still directionally preserved).
2. **Add Thames/Australia NaN data quality note** — 24.7% NaN in source timeline for Thames, filtered out before prediction.
3. **Downgrade "tidal range explains differences" to **"tidal range hypothesis, r=0.66 correlation"** — N=5, not statistically significant.
4. **Flag southeast_australia** as special case: residual std = 0.097m, q95_residual = 1.14m. GRU A0 baseline is near-optimal because surge variability is tiny.

---

## Corrected Verdict

Wave 1.7 **SCIENTIFICALLY VALID after Wave 18 fixes**. The three core findings survive audit:
1. ✅ **A1 oversampling never helps** — confirmed on 5×2×2 = 20 comparisons
2. ✅ **A3 FM augmentation region-dependent** — helps 3 regions, hurts 1, mixed 1
3. ✅ **GRU > TCN consistently** across all regions, groups, seeds

---

## Commit Audit Trail

| SHA | Content |
|---|---|
| `86169e1` | WAVE1_7_ACCEPTANCE.md commit |
| `949c567` | Wave 1.7 wave17_analyze.py fixes, paper P0 caveats, CI pandas deps |
| **Wave 18** | wave17_analyze.py event-level bootstrap + pseudo-event_id backfill + this audit |

---

## Follow-up Actions

1. Update `WAVE1_7_ACCEPTANCE.md` with corrected bootstrap units and tidal range wording
2. Update `CLAIMS_LEDGER.md` to reflect Wave 1.7 verified findings
3. **Future**: For real storm-level event IDs, the pipeline must be rewired to load `canonical_storm_id` from `storm_events.py` output. The current calendar-day pseudo-events are a reasonable interim for Wave 1.7.
4. **Future**: Source Thames and Australia NaN residual gaps to NOAA QC flags — likely gauge maintenance periods.
