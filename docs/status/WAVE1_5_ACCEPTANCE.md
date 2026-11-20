# Wave 1.5 Acceptance — TideTail

**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft)
**Head commit**: `a97ba11`
**Date**: 2026-11-01

---

## What Actually Got Done

| # | Task | Status | Evidence |
|---|---|---|---|
| P0 | CI recovery (14 ruff errors) | ✅ DONE | Commits `2d0fd24`, all 3 post-fix CI runs green (41295713437, 41296282985, 41297511274) |
| P1 | A3 full rerun (120 trainings, GPU4, Protocol F1) | ✅ DONE | `data/results/a3_flowmatching_augment_96_72.json` (120 records), save 19:20 UTC |
| P2 | Total-water evaluator module | ✅ DONE | `src/tidetail/total_water_eval.py` (233 lines, 6 functions), smoke-tested on ny_bight |
| P3 | Storm event database + dedup + event-wise split | ✅ DONE | `src/tidetail/storm_events.py` extended, 102→69 storms dedup, split 55/7/7 CLEAN |
| P4 | D0 data (ERA5, GEBCO, NOAA expansion) | ❌ BLOCKED | ERA5: no `~/.cdsapirc`; GEBCO: tile 404s; NOAA: candidates identified (115), not downloaded |
| P5 | A0-A3 formal comparison | ✅ DONE | Cross-region analysis complete, written to E12 |
| P6 | PR + ledger + paper | ✅ MOSTLY DONE | PR #23 body updated, CLAIMS_LEDGER E10/E12/E13/E14 updated, figures NOT yet generated |

## A3 Definitive Results

**120 records across 5 regions, 6 groups, 2 models, 2 seeds, Protocol F1 steps=8000. All integrity checks pass.**

### Tail95 ratio (group / A0 GRU mean) — < 1.0 = better than A0

| Region | A1/A0 | A2/A0 | A3_0.1/A0 | A3_0.25/A0 | A3_0.5/A0 |
|---|---|---|---|---|---|
| ny_bight | **0.971** | 1.005 | 1.159 | 1.321 | 1.515 |
| charleston | **0.894** | 1.157 | 1.140 | 1.602 | 1.854 |
| sydney | 1.007 | 1.127 | 1.331 | 1.694 | 2.235 |
| thames | **0.985** | 1.204 | 1.288 | 1.579 | 2.046 |
| gulf | **0.973** | 1.070 | 1.045 | 1.084 | 1.146 |
| **Global GRU mean** | **0.966** | 1.113 | 1.193 | 1.456 | 1.751 |

### Critical verdict

| Metric | Result |
|---|---|
| A3 WORSE than A0 (all 60 seed/model/region/group) | **59/60 (98.3%)** |
| A3 monotonically worsening with ratio | **5/5 regions** |
| A1 improves tail95 over A0 | **4/5 regions** |
| A1 best improvement | **charleston: -10.6%** |

**Flow Matching augmentation HURTS extreme-event forecasting.** Damage: A3_0.1 +18%, A3_0.25 +42%, A3_0.5 +71% WORSE than A0. Why FM fails: correct bulk scale (E8 confirmed) but peak-distribution mismatch → synthetic bulk residuals dilute real tail signal.

**A1 oversampling is the clear WINNER.** Directly targets tail distribution by duplicating extreme windows.

### Integrity audit (120 records)

| Check | Result |
|---|---|
| steps=8000/8000 for all records | ✅ 120/120 |
| conditioning_aligned=true (A3 only) | ✅ 60/60 |
| Single git SHA | ✅ `2d0fd24` |
| No NaN values | ✅ CLEAN |
| No duplicates | ✅ 0 |

## Storm Event Independence

| Metric | Value |
|---|---|
| Distance matches (3 gauges × IBTrACS) | 102 |
| Independent canonical storms | **69** |
| Storms hitting ≥2 gauges | 25 |
| Event-wise split train/val/test | 55 / 7 / 7 |
| Split overlaps | **0** |
| NA basin 2015-2024 unique storms | 216 |
| Tropical vs extratropical | 145 / 15 |

## CI Status

| Commit | Run | Status | ruff | pytest |
|---|---|---|---|---|
| `2d0fd24` (CI fix) | 41295713437 | ✅ | ✅ | ✅ 163 |
| `ecdb76d` (P2) | 41296282985 | ✅ | ✅ | ✅ 163 |
| `dbb7259` (P3) | 41297511274 | ✅ | ✅ | ✅ 163 |
| `a97ba11` (E12 update) | pending | — | — | — |

**Previous failing run** (41239394487, commit `a851fbf`) — 14 ruff errors — **FIXED by commit 2d0fd24**.

## Remaining Work

1. **ERA5 download** — blocked by missing CDS credentials (`~/.cdsapirc`)
2. **GEBCO bathymetry** — OPeNDAP subsetting needed (direct tiles 404)
3. **NOAA station expansion** — 115 candidates identified, not quality-filtered
4. **Total-water post-A3 reevaluation** — needs model predictions to compute true tw_tail_mae
5. **Event-level held-out forecast** — after A3 + storm splits, evaluate on canonical test storms
6. **Paper figures** — A0-A3 bar chart, storm-path/map, FM distribution comparison
7. **LaTeX compilation** — not done this wave
8. **HK region FM checkpoint** — missing, excluded from A3

## Artifact Index

| Path | Description |
|---|---|
| `data/results/a3_flowmatching_augment_96_72.json` | Definitive 120-record A3 results |
| `data/results/a3_with_totalwater.json` | Enriched with water-level threshold masks |
| `src/tidetail/total_water_eval.py` | P2 total-water evaluator (233 lines) |
| `src/tidetail/storm_events.py` | P3 storm DB + dedup + split (519 lines) |
| `paper/CLAIMS_LEDGER.md` | Updated E10/E12/E13/E14 |
| `/data/yingda/tidetail-data/raw/ibtracs/v04r01/ibtracs.since1980.list.v04r01.csv` | IBTrACS v04r01 (139MB, SHA256 verified) |
| `/data/yingda/tidetail-data/real-sources-v1/hurdat2/hurdat2-1851-2025.txt` | HURDAT2 (7.1MB, 2004 storms) |
| `/data/yingda/tidetail-data/real-sources-v1/noaa_coops/` | 360 NOAA CSVs (102MB, 3 stations) |

## Conclusion

Wave 1.5 delivered: CI recovery ✅, A3 definitive rerun (120 records, Protocol F1) ✅, total-water evaluator ✅, storm DB with event-wise splits ✅. The central negative finding — Flow Matching augmentation hurts extreme-event forecasting at all ratios — is evidence-linked across 5 regions and preserved per AGENTS.md integrity rules. A1 oversampling is the recommended tail-extreme augmentation strategy going forward.
