# WAVE 1.8 ACCEPTANCE — Event-Level Audit, Physics Documentation, Publication Figures

**Date**: 2026-11-03
**Head SHA**: `d124bcc`
**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft)
**Acceptance status**: ✅ PROVISIONAL ACCEPTANCE — P0 complete, P1-P7 partial progress, P2/P6 block GPU experiments

---

## P0 — Wave 1.7 Independent Audit ✅ COMPLETE

### Critical Discoveries

1. **Event IDs missing from .npz files** — `wave17_pipeline.py` did not pass `event_ids`/`timestamps`/`station_ids` to `save_window_predictions()`. The resulting bootstrap was window-level (overlapping windows → optimistic CI). **FIXED**: Backfilled all 190 .npz files.

2. **Wave 1.7 bootstrap was window-level, not event-level** — `paired_bootstrap()` fell back to window resampling. **FIXED**: Resamples GROUPS sharing the same pseudo_event_id (calendar day).

3. **Thames data quality** — 2296 NaN residual rows (24.7% of source timeline) from NOAA qc_flag gaps. Test sets NaN-free (filtered by pipeline). **DOCUMENTED** in WAVE17_AUDIT.md.

4. **Tidal range hypothesis** — Pearson r = 0.66 (N=5) correlation between tidal range and A3 effect. **NOT statistically significant**. Remains hypothesis, not finding.

### Corrected Event-Level Bootstrap Verdicts (n_boot=500, calendar-day pseudo-events)

| Region | Pseudo-Events | A3_better | A3_worse | Neutral | Best A3 Config | Best Δ% vs A0 GRU |
|---|---:|---:|---:|---:|---|---:|
| ny_bight | 360 | 11 | 0 | 1 | A3_fm_s99_ratio_0.5 GRU | **-15.3%** ✅ |
| us_gulf_coast | 360 | 11 | 0 | 1 | A3_fm_s42_ratio_0.5 GRU | **-12.4%** ✅ |
| south_atlantic_charleston | 360 | 12 | 0 | 0 | A3_fm_s99_ratio_0.25 GRU | **-8.9%** ✅ |
| southeast_australia | 360 | 2 | 4 | 6 | A3_fm_s99_ratio_0.5 GRU | **-0.3%** ⚠️ |
| thames_north_sea | 301 | 0 | 12 | 0 | A3_fm_s42_ratio_0.1 GRU | **+2.8%** ❌ |

### Confirmed Negative Results

- **A1 oversampling**: 0/20 comparisons (all configs × seeds × models) help. Universally harmful.
- **GRU > TCN**: Consistent across all 5 regions, 4 groups, 2 seeds per config.
- **No test-set hyperparameter selection**: All 12 A3 configs reported. "Best per region" is POST-HOCA.

### Event ID Assignment

Pseudo-event_ids derived from calendar-day grouping via:
`window_index_v1.npz → test_start_indices → all_timestamps_ns → YYYYMMDD groups`

Canonical storm IDs from `storm_events.py` are NOT wired into the pipeline yet (future work).

---

## P1 — Dataset Completion 🟡 PARTIAL

| Dataset | Status | Coverage | SHA256 | Size |
|---|---|---|---|---|
| Prime-Formal 2015-2024 | ✅ VERIFIED | 5 regions | ✓ | 451 MB |
| NOAA CO-OPS 3 stations | ✅ VERIFIED | ny_bight, us_gulf | ✓ | 102 MB |
| IBTrACS v04r01 | ✅ VERIFIED | Global | ✓ | 139 MB |
| HURDAT2 | ✅ VERIFIED | Atlantic, EPAC | ✓ | 7.1 MB |
| **ERA5** | ❌ **AUTH_REQUIRED** | Blocked by ~/.cdsapirc missing | — | ~5 GB |
| GEBCO | ❌ NOT_STARTED | Available without auth | — | ~10 GB |
| NOAA expansion | ❌ PENDING | 115 candidates | — | N/A |

**Decision**: ERA5 blocking does NOT block entire research program. Proceed with models that don't need ERA5 (standard FM, DDPM, VAE, GAN).

---

## P2 — Generative Model Benchmark ❌ NOT_STARTED

**Blocked by**: (1) GPU scheduling (GPU3 and GPU4 ARE available now), (2) need target contract verification per model.

**Immediate candidates (no ERA5 needed)**:
- Conditional VAE
- DDPM (full-budget retrain on correct residual target)
- Standard Flow Matching (10 retrained checkpoints exist as baseline)
- Tail-aware Flow Matching (use `physics.py:tail_weight()`)
- Conditional GAN / TimeGAN

**Before training**: unit tests → CPU/GPU smoke → scale check → conditioning alignment → GPU avail check → existing checkpoint check.

---

## P3 — Physics Documentation ✅ COMPLETE

`docs/methods/PHYSICS_EQUATIONS.md` written with:
1. Inverse barometer hydrostatic derivation
2. Wind stress bulk drag (Large & Pond 1981)
3. Coastal setup simplified momentum balance
4. Tail weight function definition
5. Lag kernel calibration (diagnostic baseline)
6. Total water decomposition identity
7. Planned physics-guided FM variants (tail-FM, phy-FM)
8. Formal scope and disclaimer
9. All equations map to `src/tidetail/physics.py`

---

## P4 — Publication Figures ✅ PARTIAL (6/14 = 43%)

| Fig | Title | Status | SHA |
|---|---|---|---|
| F1 | Study Regions Map | ✅ | 618482370afbe867 |
| F3 | Event Counts and Splits | ✅ | 06b628d1f8c087bb |
| F4 | Tide/Residual Decomposition | ✅ | c7e1d0b84639f358 |
| F6 | Real vs Generated Residual KDE | ✅ | ff3a2c660eacf283 |
| F8 | A0/A1/A2/A3 Performance | ✅ | c04790de96a1ddc5 |
| F9 | Tail 95/99 Performance | ✅ | fd66b2fe5629b8e7 |
| F10-F14 | Generative benchmark, physics ablation, cross-region, failure/success | ❌ PENDING | — |

All figures: vector PDF + raster PNG, 300+ DPI, derive from verified artifacts only.

---

## P5 — Manuscript 🟡 PENDING UPDATES

Current state: 18 pages, compiles, P0 caveats added 2026-11-03.

**Updates needed**:
- Add event-level CI verdicts (replace window-level numbers)
- Add Thames/Australia NaN data quality note
- Downgrade tidal range to hypothesis (r=0.66, N=5)
- Add physics equations appendix reference
- Add figure references F1-F9

---

## P6 — Unattended Execution Worker ❌ NOT_STARTED

TASK_QUEUE.json updated with correct statuses. No persistent worker connected.

---

## P7 — Deliverables ✅ PARTIAL

| Deliverable | Path | Status |
|---|---|---|
| WAVE1_8_ACCEPTANCE.md | `docs/status/WAVE1_8_ACCEPTANCE.md` | ✅ THIS FILE |
| LATEST_PROGRESS.md | `docs/status/LATEST_PROGRESS.md` | ✅ Written |
| GENERATIVE_MODEL_BENCHMARK.md | `docs/experiments/GENERATIVE_MODEL_BENCHMARK.md` | ❌ PENDING (needs P2) |
| PHYSICS_EQUATIONS.md | `docs/methods/PHYSICS_EQUATIONS.md` | ✅ Written |
| DATA_COVERAGE_MATRIX.csv | `docs/data/DATA_COVERAGE_MATRIX.csv` | ✅ Written |
| FIGURE_REGISTRY.csv | `docs/status/FIGURE_REGISTRY.csv` | ✅ Updated |
| TASK_QUEUE.json | `docs/status/TASK_QUEUE.json` | ✅ Updated |
| CLAIMS_LEDGER.md | `paper/CLAIMS_LEDGER.md` | 🟡 Needs Wave 1.7 updates |
| paper/main.tex | `paper/main.tex` | 🟡 Needs P5 updates |

---

## Verification Checklist

| Item | Check | Result |
|---|---|---|
| Prediction files | 190/190 exist with event_ids + timestamps | ✅ |
| FM checkpoints | 10 retrained 2026-11-03 21:27-21:47 | ✅ |
| Pytest | 270 pass, 0 fail, 51s | ✅ |
| Physics module | 50 tests, hPa→Pa verified | ✅ |
| Event counts | ny_bight=360, thames=301, us_gulf=360, charleston=360, aus=360 | ✅ |
| Bootstrap method | Event-level group resampling (calendar-day pseudo-events) | ✅ |
| Paper compilation | Not yet verified post Wave 1.8 updates | ❓ |

---

## Next Priorities (Ordered)

1. **GPU start tail-aware FM training** (GPU3+GPU4 both free, ~24 GB each)
2. **Download GEBCO regional subsets** (no auth needed, ~10 GB)
3. **Run generative benchmark** (DDPM retrain + VAE + conditional FM variants)
4. **Generate remaining figures** (F10-F14 from P2 results)
5. **Update manuscript** with Wave 1.7 audit corrections and figure references
6. **Update CLAIMS_LEDGER.md** with verified event-level findings
7. **Implement supervised task worker**

---

## Commit Trail

| SHA | Content |
|---|---|
| `86169e1` | WAVE1_7_ACCEPTANCE.md — original |
| `949c567` | wave17_analyze.py fixes, paper P0 caveats, CI deps |
| `0468a18` | Wave 18 — event-level bootstrap, event_id backfill, independent audit |
| `d124bcc` | Resolve TASK_QUEUE rebase, mark W17-T1 COMPLETE, scientific_gate RESOLVED |

---

## Caveats Preserved

1. Pseudo-event_ids are calendar-day groups, not canonical storm IDs
2. Thames NaN gaps (24.7% source) but test set NaN-free
3. A3 "best per region" is POST-HOCA, not unbiased
4. A1 oversampling negative result preserved (0/20)
5. Tidal range hypothesis NOT established (r=0.66, N=5 not significant)
6. ERA5 physics priors blocked by auth
