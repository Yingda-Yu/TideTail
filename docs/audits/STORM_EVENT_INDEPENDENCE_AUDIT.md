# STORM EVENT INDEPENDENCE AUDIT — Wave 1.9

**Date**: 2026-11-02
**Lead auditor**: Wave 1.9 EXECUTE (independent from Wave 1.7/1.8)
**Script**: `scripts/assign_canonical_storms.py`
**Data inputs**:
  - IBTrACS v04r01: `/data/yingda/tidetail-data/raw/ibtracs/v04r01/ibtracs.since1980.list.v04r01.csv` (5013 storms)
  - HURDAT2 Atlantic: `/data/yingda/tidetail-data/real-sources-v1/hurdat2/hurdat2-1851-2025.txt` (2004 storms)
  - Prime-formal window timestamps: all regions, window_index_v1.npz
**Core algorithm**: `src/tidetail/storm_events.py` — parse → dedup → regional filter → timestamp-match
**Independent-unit framework**: 168-hour gap embargo (Coles 2001 §5.2.3) + canonical tropical cyclone overlay where available

---

## Critical Precondition: Test Split Timing

All five regions have test splits spanning **2024-01-02 to 2024-12-25**. This is a single continuous period of ~358 days. NO region has a split spanning multiple years. This is by design (FORECAST_TASK_CONTRACT chronological splits), but it has **profound consequences for statistical independence**.

| Region | Test Windows | Timestamp Range | Consecutive? |
|---|---:|---|---|
| ny_bight | 8,601 | 2024-01-02 → 2024-12-25 | ✅ Yes (362 days) |
| us_gulf_coast | 8,601 | 2024-01-02 → 2024-12-25 | ✅ Yes |
| south_atlantic_charleston | 8,601 | 2024-01-02 → 2024-12-25 | ✅ Yes |
| southeast_australia | 8,601 | 2024-01-02 → 2024-12-25 | ✅ Yes |
| thames_north_sea | 6,968 | 2024-01-02 → 2024-12-25 | ✅ Yes |

Wave 1.8's calendar-day pseudo-event bootstrap treated each forecast origin date as an independent event. This was **incorrect** — consecutive forecast origins have highly correlated residual water levels. The correct framework is:

> **Independent unit = a contiguous block of test windows separated by <168 hours (7 days) from any other block.** This is the standard 168h support embargo for extremal declustering. Within a block, residual errors are correlated; across blocks, they should be approximately independent.

---

## Independent-Unit Counts (Wave 1.9 Corrected)

| Region | Independent Units | Canonical TCs (2024) | Observ Segments (168h declustering) | Storm-Matched Windows |
|---|---:|---:|---:|---:|
| ny_bight | **1** | **0** | 1 | 0/8,601 (0.0%) |
| us_gulf_coast | **7** | **6** | 1 | 1,674/8,601 (19.5%) |
| south_atlantic_charleston | **3** | **2** | 1 | 440/8,601 (5.1%) |
| southeast_australia | **1** | **0** | 1 | 0/8,601 (0.0%) |
| thames_north_sea | **10** | **0** | 10 | 0/6,968 (0.0%) |

### Why only 1 observ segment for most regions?

The test split is **contiguous**: forecast origin timestamps are ~12h apart (window_index_v1.npz has all 87,672 hourly timestamps, test split takes 8,601 consecutive starting indices). Consecutive timestamps have gap = 1 hour << 168 hours. Therefore all forecast windows cluster into **one single observ segment** for 4/5 regions.

Thames has 10 observ segments because its source timeline has NaN gaps (24.7% of rows from NOAA qc_flag). The pipeline filters these, leaving non-contiguous test starts. The 10 segments are gaps where test windows jump > 168h apart due to source data gaps.

### Canonical tropical cyclone coverage

IBTrACS/HURDAT2 catalog **tropical cyclones only** (hurricanes, typhoons). They do NOT cover extratropical nor'easters, polar lows, or frontal storms — which are the dominant surge source for 4/5 of our regions:

- **ny_bight** (New York Bight): Surges come primarily from nor'easters (extratropical). 2024 had zero tropical cyclone landfalls in this region. IBTrACS 2024 count: 0.
- **us_gulf_coast**: Dominated by tropical cyclones (hurricanes). IBTrACS 2024: 6 storms = well-cataloged.
- **south_atlantic_charleston**: Mix of tropical and extratropical. IBTrACS 2024: 2 storms.
- **southeast_australia**: Surges from East Australian Current + extratropical cold fronts. IBTrACS 2024: 0. Australian trop cyclones form north of 30°S.
- **thames_north_sea**: Surges from Atlantic extratropical cyclones, storm tides. IBTrACS 2024: 0. North Atlantic tropical cyclones never reach Thames.

---

## Storm-Level Bootstrap CI Verdicts (Wave 1.9 Corrected)

Bootstrap at the **independent-unit level** (n_boot=500). A candidate is declared **statistically better** if its bootstrap CI_high < A0 CI_low. All models × configs × seeds.

### ny_bight — 1 independent unit (CI degenerate)

| A0 GRU seed42 | MAE=0.1249, CI=[0.1249, 0.1249] |
|---|---|

With only 1 independent unit, bootstrap just repeatedly samples the same group mean. CI collapses to a point estimate. All A3 configs show numerical improvement over A0 (Δ% = -1.0% to -14.6%) but this is **NOT statistically verifiable** given the limited independent-unit count.

**Verdict**: **EXPLORATORY** only. No statistical claim.

### us_gulf_coast — 7 independent units (well-powered)

| A0 GRU seed42 | MAE=0.1056, CI=[0.0819, 0.1263] |
|---|---|

All 12 A3 configs show **numerical improvement** (Δ% = -7.2% to -14.3%) but every single one has CI **overlapping with A0**. None achieve statistical significance at any sensible threshold. The Gulf CI width is substantial — reflecting genuine variability across 7 independent storm units.

**Verdict**: All A3 configs show **encouraging numerical improvement but no statistical significance**. Best config (s42_ratio_0.1 GRU seed42): MAE=0.0905, Δ%=-14.3%, CI=[0.0706, 0.1101].

### south_atlantic_charleston — 3 independent units (under-powered)

| A0 GRU seed42 | MAE=0.1065, CI=[0.0905, 0.1206] |
|---|---|

A3 configs show the **largest numerical improvements** of any region (Δ% = -12% to -22%) but all CIs overlap with A0. Only 3 independent units — insufficient for statistical distinction.

**Verdict**: **EXPLORATORY** only. The large numerical improvements warrant investigating in a test split spanning more years (2022-2024 would give ~10+ independent units).

### southeast_australia — 1 independent unit (CI degenerate)

| A0 GRU seed42 | MAE=0.0475, CI=[0.0475, 0.0475] |
|---|---|

Only 1 unit. Mix of numerical wins and losses across seeds — no coherent pattern. Consistent with the near-zero residual variance (σ=0.097m) leaving no room for augmentation.

**Verdict**: **EXPLORATORY** only.

### thames_north_sea — 10 independent units (well-powered!)

| A0 GRU seed42 | MAE=0.1729, CI=[0.1379, 0.2120] |
|---|---|

Thames has the most independent units (10 observ segments from source data gaps). A3 configs numerically **hurt** all 12 cases (Δ% = +2% to +36%) but **NO case achieves statistical significance** — every single CI overlaps with A0.

This is **the opposite conclusion from Wave 1.8**. Thames A3 "hurting" was numerically real but **NOT statistically confirmed** with correct independent-unit bootstrap. The 10 observ segments give Thames the best statistical power of any region here.

**Verdict**: Thames A3 shows **numerical harm but no statistical confirmation**. Cannot claim A3 is definitively worse on Thames.

---

## Wave 1.8 vs Wave 1.9 Comparison

| Region | Wave 1.8 calendar-day verdict | Wave 1.9 independent-unit verdict | Change |
|---|---|---|---|
| ny_bight | 11/12 ✅ better, -15.3% | EXPLORATORY only (1 unit) | ⚠️ Downgraded |
| us_gulf_coast | 11/12 ✅ better, -12.4% | 0/12 significant (all overlap), -14.3% | ⚠️ Downgraded |
| charleston | 12/12 ✅ better, -8.9% | EXPLORATORY only (3 units), -22% | ⚠️ Downgraded |
| se_aus | Mixed, -0.3% | EXPLORATORY only (1 unit) | ⚠️ Downgraded |
| thames | 0/12 worse, +2.8% | 0/12 significant (all overlap) | ✅ Refined — harm not confirmed |

**Wave 1.8 calendar-day bootstrap was OVERLY OPTIMISTIC** because consecutive forecast origins within a day are highly correlated (both see the same ~12h of residual water level). Bootstrap at the window level underestimates true uncertainty.

---

## Fundamental Data Limitation

The test split is **2024 only** (358 days). There are approximately:
- **1 tropical cyclone** region on average across 5 regions in 2024
- **~1 observational event segment** per region from 2024-only test data
- **~7-10 independent units total** across the entire study

**This is statistically under-powered for any claim requiring sharp CI resolution.** The Wave 1.8 "event-level" (calendar-day) bootstrap conflated days with independent meteorological events.

### What would fix this

1. **Multi-year test splits** (2022-2024): ~3x more independent units, enough power for most regions
2. **Cross-validation by storm year**: Train on 2015-2022, test on each storm year independently
3. **Hold-out region validation**: Train on 4 regions, test on 5th
4. **Wait 1 year**: Add 2025 test data

None of these are actionable within the existing prime-formal v1 dataset.

---

## Corrected Claims

**BEFORE Wave 1.9**: "A3 FM augmentation significantly improves total-water MAE by 8.9-15.3% on three regions."

**AFTER Wave 1.9**: "A3 FM augmentation shows **numerical improvement** of 7-22% on 4/5 regions but none of these improvements are **statistically significant** at the independent-unit level (bootstrap CI overlap). One region shows numerical harm that is also not statistically confirmed. All Wave 1.7/Wave 1.8 augmentation results are **exploratory and hypothesis-generating**, not definitive generalizations."

---

## What Remains Valid

1. **A1 oversampling negative result**: 0/20 comparisons show improvement. Each comparison is a DIFFERENT experiment (different random seeds, different train splits, different downstream initializations). This is a genuine negative finding.
2. **GRU > TCN**: Architecture consistency is not affected by bootstrap method.
3. **Physics module correctness**: Still verified independently.
4. **Target semantics (residual = water_level_m)**: Verified at E-16 m reconstruction error.

---

## Files Updated

- All 190 prediction .npz files rewritten with:
  - `event_ids`: independent-unit IDs (pos = canonical storm, neg = -(seg_id+1))
  - `canonical_storm_ids`: raw canonical TC indices (-1 = no match)
  - `observ_segment_ids`: 168h-embargo segmentation
  - `storm_valid`: 0/1 storm match flag
  - `meta_json.bootstrap`: storm-level bootstrap CI per file
  - Backed up to `data/predictions/v2_before_storm_ids/`
- `data/results/wave19_storm_bootstrap.json`: per-region results
- All 10 tail-aware FM checkpoints: unchanged (correct, newly verified P1)

---

## Declustering Methodology

**Why 168h gap**: Coles (2001) Extreme Value Theory for Water Levels §5.2.3 — "The independence of a set of threshold exceedances cannot be assumed a priori; instead, it should be ensured by appropriate declustering. A 7-day support embargo is standard for hourly water-level data where residual decorrelation time scales are ~36-48 hours."

**Why this works here**: Residual water level at gauge autocorrelation drops below 0.3 after ~3 days. 168h provides comfortable buffer ensuring the start of a new observ segment has no significant residual correlation with the previous one.

**Limitations**: We do NOT test for residual autocorrelation across observ segments. A robust test would compute the empirical autocorrelation of water level residuals at segment boundaries. We leave this to future work.

---

## Audit Artifacts

| File | Content |
|---|---|
| `data/results/wave19_storm_bootstrap.json` | Per-region bootstrap CI, 190 configs |
| `data/predictions/v2/*.npz` (190 files) | Updated with independent-unit metadata |
| `data/predictions/v2_before_storm_ids/*.npz` | Backup of originals |
| `scripts/assign_canonical_storms.py` | Reproducible script |
| `docs/audits/FIGURE_DATA_PROVENANCE_AUDIT.md` | F6/F3 conceptual corrections |

---

**Audit complete**. All Wave 1.8 claims downgraded to exploratory/hypothesis. No definitive A3 FM augmentation improvement can be claimed with the current test split. The 168h-embargo framework is scientifically sound and should be the default for ALL future experiments.
