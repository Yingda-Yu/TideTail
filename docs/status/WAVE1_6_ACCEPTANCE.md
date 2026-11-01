# WAVE 1.6 ACCEPTANCE — Critical Audit and Scientific Recovery

**Date**: 2026-11-03
**Branch**: `trae/global-data-generative-research-20261031`
**Baseline SHA**: `a288c2b` (WAVE1_5 final)
**New commits (3)**:
| SHA | What |
|---|---|
| `fc45703` | P0: resolve double-detiding target semantics across all loaders + 55 regression tests + audit doc |
| `ad63c1b` | P2: repair total_water_eval.py — event-resampled bootstrap + peak split + prediction save/load |
| `e88036f` | P3: storm_events — IBTrACS csv module + HURDAT2 proper parse + STATUS-based typing |

**CI**: ✅ All local tests pass (218 pytest, 1 warning, ruff clean, 65.7s)
**PR**: #23 (Draft, branch pushed, DO NOT merge main)

---

## Critical Finding — P0 Double-Detiding

**Confirmed 100% across all 5 prime regions**:

```
observed_total_water_level_m = water_level_m + tide_level_m   (MAE = 0.00000000m, r = 1.0)
```

`water_level_m` IS ALREADY the de-tided surge residual per FORECAST_TASK_CONTRACT.md §2 line 29.

Both `train_generative_models.py` and (pre-11-01) `run_a3_flowmatching_augment.py` computed `residual = water_level_m - tide_level_m` — **a double-subtraction** that added an oscillatory tidal component (~0.5m std) to the surge residual.

### Damage Inventory

| Item | Target used | Status |
|---|---|---|
| **A0 baselines** (`run_a0_baselines.py`) | `water_level_m` correctly | ✅ **UNAFFECTED** — sole valid historical baseline |
| **Pilot MLP** (`pilot_data.py`) | `water_level_m` correctly | ✅ **Unaffected** |
| **M2 Diffusion-TS** (`train_generative_models.py`) | Buggy — "full water" path used `water_level_m` (residual!) instead of `observed_total` | ❌ **Damage documented** (audit §3.2) |
| **M4 Flow Matching** (`train_generative_models.py`) | Buggy — `water_level - tide_level` | ❌ **Targets corrupted** |
| **M3/M5/M6** | Inherited same loader bug | ❌ **Engineering-damaged** |
| **FM checkpoints** (10 files) | Trained on double-detided residuals | ❌ **INVALIDATED — need retrain** |
| **A3 run** (120 records) | FM checkpoint + downstream on corrupt targets | ❌ **Marked EXPLORATORY** |

### Corrected Scripts

| Script | Before | After | Verified |
|---|---|---|---|
| `scripts/train_generative_models.py` | `residual = water - tide` | `residual = water` (already residual). Diffusion-TS path uses `observed_total_water_level_m`. | `test_train_generative_model_loader_patches_residual` + `test_train_generative_model_loader_patches_total_water` (5 regions each) |
| `scripts/run_a3_flowmatching_augment.py` | Already patched 2026-11-01 | Verified against data, ruff clean | `test_train_generative_model_loader_patches_residual` |
| `scripts/run_a0_baselines.py` | Always read `water_level_m` correctly | No change needed | `test_a0_baselines_loader_patches_residual` (5 regions) |

### Regression Tests Locking the Contract

New file: `tests/test_target_semantics_contract.py` — **55 tests across 5 regions**, runs against REAL data every time CI runs. If the double-subtraction bug ever returns, CI will fail immediately.

---

## P2 Total-Water Evaluation — Repaired

Rewrote `src/tidetail/total_water_eval.py` (153 → 329 lines, 6 → 7 functions):

| Fixed issue | Before | After |
|---|---|---|
| **Bootstrap resampling unit** | Resampled OVERLAPPING WINDOWS (inflated significance) | Resamples INDEPENDENT EVENT IDs; windows from same storm always stay together |
| **Peak metrics conflation** | Single `tw_peak_mae` mixing magnitude and timing | THREE metrics: `tw_peak_mag_mae` (compare peak heights), `tw_err_at_obs_peak_mae` (error at observed peak time), `tw_peak_timing_mae_hours` |
| **Delta semantics** | Ambiguous `positive_better`/`negative_better` | `delta = metric(A) - metric(B)` with explicit flags: `candidate_better` (CI entirely negative = A better) / `baseline_better` / `neutral` |
| **Prediction persistence** | A3 saved only aggregate metrics; P5 reruns would need full retrain | `save_window_predictions()` / `load_window_predictions()` — saves per-window residuals, tide, ground truth, timestamps, event/station IDs, git SHA, checkpoint paths |
| **Datum/unit clarity** | Implicit `metres` | Explicit `DATUM = "MLLW"`, `units = "metres"` constants; docstring states physical bounds |
| **Event + station outputs** | Only event-level | Added `station_means` to `event_level_metrics()` |

Verified on real prime data: reconstruction MAE ≤ 1e-8m across all 5 regions.

---

## P3 Storm Events — Repaired

| File | Before | After |
|---|---|---|
| IBTrACS loader | `split(",")` — fragile for quoted fields | `csv.reader()` — standards-compliant, handles all quoting |
| HURDAT2 loader | Wrong column positions, `split(",")`, no lat/lon parsing, no date format | Full rewrite — parses YYYYMMDD/HHMM dates, lat/lon with N/S/E/W suffixes, STATUS-based typing (HU→cyclone, TS→storm, ET→extratropical), loads 55,605 rows → 2,004 Storm objects |
| Storm typing | Latitude + peak wind heuristic only | Uses STATUS field (HU/TS/ET/TD); falls back to peak wind if no status |
| `StormGaugeMatch.time_offset_hours` | Name misleading (was "peak_time - gauge" but computed "peak - nearest_track") | Renamed to `storm_peak_to_nearest_track_hours` + `validity_tier` field |

New: `parse_hurdat2_to_storms()` — HURDAT2 rows → Storm objects (2,004 storms loaded).

---

## P1 A3 Negative Result — Evidence-Bounded

**Preserved** (NOT deleted): `data/results/a3_flowmatching_augment_96_72.json` (120 records, all integrity checks pass).

**Evidence-bounded verdict**:
```
EXPLORATORY NEGATIVE RESULT for the tested Flow Matching augmentation
implementation under damaged target semantics.

Does NOT generalize to the Flow Matching model family as a whole.
```

Why: The FM checkpoint and downstream forecasters were both trained on the double-detided target (`water - tide`). The 59/60 (98.3%) A3 worse-than-A0 finding is internally consistent (all groups used the same wrong target) but **absolute values are NOT scientifically meaningful**. P5 rerun required.

## P4 Data Acquisition — Blocked (Credentials)

| Item | Status | Owner Action Required |
|---|---|---|
| **ERA5 CDS API** | Need `~/.cdsapirc` with personal access token | Create at https://cds.climate.copernicus.eu/user/ → get token → `echo "url: https://cds.climate.copernicus.eu/api\nkey: YOUR-TOKEN" > ~/.cdsapirc` |
| **GEBCO 2026** | Can proceed without auth — regional subsets from `download.gebco.net` or CEDA OPeNDAP | User action preferred; tech can queue download |
| **NOAA expansion** | Data available, quality-filter needed (115 candidates → need 10+ years + datum + 2015-2024) | Queue behind P5 pilot |

## P5 Controlled Recovery — Blocked (GPU)

**Requires GPU time on non-GPU0**:
1. Retrain 10 FM checkpoints on correct residual target
2. Retrain A0/A1/A2/A3 downstream forecasters on correct residual target
3. Save per-window predictions via new `save_window_predictions()`
4. Evaluate on TOTAL WATER via repaired `evaluate_total_water()`
5. Paired event-level bootstrap CI via `paired_event_bootstrap_ci()`
6. Only expand to 5 regions when 1-region pilot passes

**GPU queue**: Check `nvidia-smi` before starting. GPU0 NEVER. Preferred: GPU4 or GPU5.

## P6 Physics-Guided Models — Deferred

Wait for P5 pilot to establish baseline with correct targets before launching physics-guided variants:
- Conditional Flow Matching + U10/V10/MSLP
- Conditional FM + inverse-barometer prior
- Conditional FM + coastal wind-stress prior
- Tail-weighted FM
- Peak/duration-constrained FM
- TimeGrad, proper Diffusion-TS, TimeGAN, VAE, GP-prior FM

Do NOT claim shallow-water PDE conservation from single-gauge observations alone.

---

## CLAIMS_LEDGER Changes

| ID | Change |
|---|---|
| **E7** (Diffusion-TS "fundamentally broken") | Added P0 note: loader used residual instead of total water; retraining needed before conclusion |
| **E8** (Flow Matching "correct scale") | **INVALIDATED** — trained on double-detided residuals; all 10 checkpoints need retrain |
| **E12** (A3 "Flow Matching hurts tail95") | **RE-CLASSIFIED EXPLORATORY** — both FM and downstream forecasters trained on wrong target |
| **E13** (P2 total-water eval) | **REPAIRED** — rewritten after P0 audit; 4 key issues fixed (bootstrap, peak, delta, save/load) |
| **E14** (P3 storm DB) | New parse_hurdat2_to_storms(), HURDAT2 STATUS-based typing; validity_tier added |
| **E9** (A0 baselines) | No change — UNAFFECTED (read water_level_m directly) |

---

## What Preserved / What Deleted

**Preserved (NOT deleted)**:
- `data/results/a3_flowmatching_augment_96_72.json` — marked EXPLORATORY
- `data/results/flow_matching_96_72.json` — marked INVALIDATED
- `data/results/diffusion_ts_96_72.json` — marked INVALIDATED (note: loader also wrong)
- `data/checkpoints/flow_matching_*.pt` — marked INVALIDATED

**Never existed**: No experiments were overwritten or silently deleted.

---

## What Remains for Next Session

| Priority | Task | Blocked by |
|---|---|---|
| **P5** (HIGH) | Retrain FM on correct residual → rerun A3 with P2 total-water eval | GPU (need non-GPU0) |
| **P6** | Physics-guided FM variants | P5 pass |
| **P4** | ERA5 CDS auth + GEBCO regional subsets | Credentials (user action) |
| **P7 paper** | LaTeX compile + PR #23 body update | P5 results |
| Graphs | A0-A3 tail-error, event-count, FM-tail-distribution, data-availability | P5 results |

---

## Scientific Takeaway

```
BEFORE WAVE 1.6:
  "Flow Matching hurts tail95 by 98.3% — definitive result"

AFTER WAVE 1.6:
  All historical generative-model runs are INVALIDATED — they
  trained on DOUBLE-SUBTRACTED residuals, not the surge signal.
  
  A0 baselines remain the ONLY valid historical baseline
  (they always read water_level_m correctly).
  
  P5 rerun required with corrected targets before any scientific claim
  about Flow Matching augmentation can be made.
  
  The negative A3 finding is preserved as EXPLORATORY: it showed the
  tested implementation with the double-detided target had this ordering.
  It does NOT imply Flow Matching or generative augmentation in general
  is ineffective for tide-surge forecasting.
```

---

## Commit Chain

```
fc45703  fix: P0 CRITICAL — resolve double-detiding target semantics
ad63c1b  feat: P2 repair total_water_eval.py — event bootstrap, peak split, save/load
e88036f  feat: P3 storm_events — csv module, HURDAT2 proper parse, status typing
```

## Integrity Check

| Check | Result |
|---|---|
| `git diff --check` | ✅ No whitespace issues |
| `ruff check src/ tests/ scripts/` | ✅ All checks passed |
| `pytest tests/ -q` | ✅ 218 passed, 1 warning |
| `pytest tests/test_target_semantics_contract.py -v` | ✅ 55 passed (real data) |
| Contract verified per region | ✅ `w + tide = total` MAE < 1e-6m in all 5 |

---

**Acceptance criterion**: Wave 1.6 passes all local tests, all P0/P2/P3 code is correct against real data, damage inventory is complete, preserved results are clearly marked EXPLORATORY/INVALIDATED, no claim overstates evidence.

**Status**: ✅ PASS — next session begins P5 controlled recovery.
