# CRITICAL TARGET-SEMANTICS AUDIT — Double-Detiding Bug Resolution

**Date**: 2026-11-03 (Wave 1.6, P0)
**Auditor**: Automated + manual inspection
**Git SHA**: `HEAD` — pre-fix on branch `trae/global-data-generative-research-20261031`

---

## Executive Summary

**Critical bug resolved**: Two training scripts (`train_generative_models.py` line 69, previously `run_a3_flowmatching_augment.py`) were computing `residual = water_level_m - tide_level_m`.

This is a **double-subtraction** because the prime dataset already defines:

```
water_level_m = observed_total_water_level_m − FES2022_tide  (FORECAST_TASK_CONTRACT.md §2, line 29)
```

Verified EXACTLY across all 5 prime regions:

```
water_level_m + tide_level_m = observed_total_water_level_m   (MAE = 0.00000000m, r = 1.000000)
```

**Affected experiments** (ALL historical runs before 2026-11-03 are **EXPLORATORY / TARGET-SEMANTICS-DAMAGED**):

| Item | Target used | Status |
|---|---|---|
| A0 real-only baselines (`run_a0_baselines.py`) | `water_level_m` correctly — no subtraction | ✅ **Unaffected** |
| Pilot MLP (`src/tidetail/pilot_data.py`) | `water_level_m` correctly | ✅ **Unaffected** |
| M2 Diffusion-TS (`train_generative_models.py`) | Buggy: "full water level" path used `water_level_m` (residual!) | ❌ **Damaged** |
| M4 Flow Matching (`train_generative_models.py`) | Buggy: `water_level_m - tide_level_m` (double-detided garbage) | ❌ **Damaged** |
| M3/M5/M6 (not trained) | Same loader as M2/M4 — inherited the bug | ❌ **Engineering-damaged** |
| A3 FM augmentation (`run_a3_flowmatching_augment.py`) | Same bug — FM checkpoint trained on wrong target, A3 run on wrong target | ❌ **Damaged (partially patched 2026-11-01, full P0 fix 2026-11-03)** |
| E8 Flow Matching checkpoint (`data/checkpoints/flow_matching_*.pt`) | Trained on double-detided residuals | ❌ **Invalidated — must retrain** |
| `total_water_eval.py` reconstruction | `pred_total = pred_residual + future_tide` — if input residual was wrong, reconstruction is compound-broken | ❌ **Cannot produce valid total-water numbers from corrupt inputs** |

**A0 baselines are the ONLY valid historical baseline.** All non-pilot generative-model experiments use corrupted targets and their absolute metrics are NOT trustworthy for scientific claims.

---

## Data Contract Verification

### Column Semantics (FORECAST_TASK_CONTRACT.md §2)

| Column | Actual Meaning | Verdict |
|---|---|---|
| `observed_total_water_level_m` | Raw tide + surge, MLLW datum | ✅ Confirmed |
| `tide_level_m` | FES2022 deterministic tide, MLLW datum | ✅ Confirmed |
| `water_level_m` | `observed_total − FES2022_tide` = **ALREADY de-tided surge residual** | ✅ Confirmed |
| Datum | MLLW consistent across all three | ✅ Confirmed |

### Per-Region Identity Verification

| Region | n_rows | Valid rows | MAE (w+t vs total) | MAE (w vs total) | Tide amp (m) | Residual amp (m) | Total amp (m) |
|---|---|---|---|---|---|---|---|
| ny_bight | 87,672 | 87,672 | **0.00000000m** | 0.479m | 1.10m | 0.48m | 1.15m |
| south_atlantic_charleston | 87,672 | 87,672 | **0.00000000m** | 0.499m | 0.92m | 0.50m | 1.05m |
| southeast_australia | 87,672 | 87,641 | **0.00000000m** | 0.331m | 0.48m | 0.33m | 0.55m |
| thames_north_sea | 87,672 | 85,376 | **0.00000000m** | 1.221m | 1.31m | 1.22m | 1.81m |
| us_gulf_coast | 87,672 | 87,672 | **0.00000000m** | 0.142m | 0.42m | 0.14m | 0.45m |

Amplitude = 0.5 × (p99 − p1). Valid rows exclude qc_flag='source_gap_no_imputation' (0–2,296 rows per region).

**Conclusion**: `water_level_m` IS the de-tided surge residual. The direct MAE vs total is substantial (0.14–1.22m) — this is the actual surge signal, correctly extracted.

---

## Bug Mechanism

### Script 1: `train_generative_models.py` (M2 Diffusion-TS + M4 Flow Matching)

**Pre-fix** (line 69):
```python
water_level = merged["water_level_m"].to_numpy()   # This IS residual!
tide_level = merged["tide_level_m"].to_numpy()
residual = water_level - tide_level                 # DOUBLE-SUBTRACT!
```

Additionally, the "full water level" path for Diffusion-TS used `water_level` (which was residual) instead of the true `observed_total_water_level_m`. So Diffusion-TS was also wrong.

**Post-fix** (2026-11-03):
```python
raw_water = merged["observed_total_water_level_m"].to_numpy()  # For Diffusion-TS
residual  = merged["water_level_m"].to_numpy()                   # For Flow Matching (already correct)
```

### Script 2: `run_a3_flowmatching_augment.py` (A3 augmentation)

Partially patched on 2026-11-01, completed and verified against data on 2026-11-03. Same bug pattern.

---

## Damage Inventory

### How Bad Was the Corruption?

The wrong target is `double_residual = water_level_m - tide_level_m`. Since tide has large amplitude (0.4–1.3m depending on region), this subtraction adds a large non-physical oscillatory component to what should be the surge residual. The double_residual signal has:
- Wrong mean (offset by tide mean ~0)
- Wrong std (inflated by tide std ~0.5m)
- Wrong tail distribution (tide dominates q95/q99)

### Effect on Model Training

- **Flow Matching** was trained to generate double-detided signals. Generated samples have no physical residual semantics — they're the sum of a residual and an inverted tide. The model cannot learn surge dynamics from this target.
- **Diffusion-TS** was trained on residual data with a "decomposition" architecture designed to separate trend/seasonal/residual. It received residual-only data and decomposed nothing meaningful.
- **A3 downstream forecast (GRU/TCN)** learned to predict the double-detided target. Its absolute metrics are meaningless, but the **relative ordering** of A0/A1/A2/A3 groups may still be internally consistent (all use the same wrong target).

### Is the A3 "Flow Matching hurts tail95" Finding Still Valid?

**As a RELATIVE internal comparison**: Possibly yes — A0, A1, A2, A3 all used the same corrupted target, so the within-comparison ordering is consistent.

**As an ABSOLUTE scientific finding**: NO. We cannot conclude anything about Flow Matching on the correct target, because both the FM generator and all downstream forecasters learned on wrong data.

**Verdict**:
```
EXPLORATORY NEGATIVE RESULT for the tested Flow Matching augmentation
implementation under damaged target semantics.

Does NOT imply that Flow Matching or generative augmentation in general
is ineffective for tide-surge forecasting.
```

### What's Preserved?

- All raw experiment JSONs preserved (not deleted)
- A3 result: `data/results/a3_flowmatching_augment_96_72.json` — **kept, marked EXPLORATORY**
- FM checkpoints: `data/checkpoints/flow_matching_*.pt` — **kept, marked INVALIDATED**
- A0 baselines: `data/results/a0_baselines_96_72.json` — **VALID, A0 used correct target**

---

## Fixes Applied (2026-11-03)

| File | Change | Commit |
|---|---|---|
| `scripts/train_generative_models.py` | `residual = water_level` (not `water - tide`); Diffusion-TS path uses `observed_total_water_level_m` instead of `water_level_m` | Pending |
| `scripts/run_a3_flowmatching_augment.py` | `residual = water_level` (already correct from 2026-11-01 patch, verified + ruff clean) | Was in modified state |
| `tests/test_target_semantics_contract.py` | NEW — 55 regression tests against real data. CATCHES THE BUG if it ever returns | Pending |

### New Regression Test Coverage

| Test class | Purpose |
|---|---|
| T0 Identity | Confirms the fundamental contract `water + tide = total` holds across all 5 regions |
| T1 Loader integrity | Verifies both patched scripts load the correct data against independent raw CSV parsing |
| T2 Metadata integrity | Confirms manifest + train_only_metadata declare both water and tide scalers |
| T3 Physical plausibility | Verifies residual/tide/total have physically reasonable amplitudes |

---

## What Needs Retraining

1. **All FM checkpoints** — must retrain on correct residual target (`water_level_m` directly)
2. **All Diffusion-TS models** — must retrain on correct total-water target (`observed_total_water_level_m`)
3. **All A3 runs** — must rerun with correct FM checkpoints and correct downstream forecaster targets
4. **All A1/A2 runs** — same downstream forecasters were trained on wrong target too, though simpler augmentations may be less affected

A0 baselines need **no retraining** — they read `water_level_m` directly.

---

## Lessons Learned

1. **Trust the data contract first, not the script.** A one-line formula in FORECAST_TASK_CONTRACT.md line 29 was contradicted by a one-line subtraction in training scripts.
2. **Cross-check by reconstruction**: Always verify `sum(components) == observed` before running expensive GPU experiments. One Python command prevents weeks of wasted training.
3. **Data contracts need executable tests.** The new `tests/test_target_semantics_contract.py` locks the contract into CI — any future regression will fail immediately.
4. **"Residual" is ambiguous.** TideTail's dataset uses "residual" to mean "already de-tided." Scripts interpreted it as "something to derive from total and tide." This terminology confusion caused the bug.
