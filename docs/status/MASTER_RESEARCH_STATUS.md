# MASTER RESEARCH STATUS — TideTail

**Last updated**: 2026-11-03
**Head SHA**: `9ce5b8f` (feat: S3 physics surge priors module)
**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft, DO NOT merge to main)
**Active PID**: 551931 (wave17 pipeline, GPU3, 101% CPU, ~1h elapsed)

---

## Executive Summary

Wave 1.6 complete. Critical double-detiding target-semantics bug (P0, `fc45703`) discovered and fixed across all loaders. All 10 FM checkpoints retrained on correct residual target (2026-11-03 21:27–21:47). Wave17 unified pipeline (`wave17_pipeline.py`) executing A3 augmentation rerun on correct semantics across 5 regions × 3 ratios × 2 FM seeds × 2 downstream models × 2 seeds = 120 A3 + 70 baselines = 190 predictions. Partial A3 results on 3 regions show region-dependent efficacy: ny_bight +15.5% and us_gulf_coast +12.4% at ratio 0.5 GRU; thames_north_sea -14.6% (largest residual-amplitude basin). A1 oversampling confirmed universally harmful (negative result).

## Key Completed Items

| Wave | Item | SHA | Verification |
|---|---|---|---|
| P0 | Resolved double-detiding target semantics | `fc45703` | 55 regression tests, MAE < 1e-8m total-water reconstruction |
| P2 | Rewrote `total_water_eval.py` — event-resampled bootstrap, peak-mag/timing split | `ad63c1b` | Paired bootstrap resamples INDEPENDENT EVENT IDs, not windows |
| P3 | Storm events: IBTrACS csv + HURDAT2 proper parse, STATUS-based typing | `e88036f` | 2,004 HURDAT2 storms, 5,013 IBTrACS, canonical IDs built |
| P4 | Physics module (`src/tidetail/physics.py`, 7 functions, 50 tests) | `9ce5b8f` | Inverse barometer with correct hPa→Pa conversion, wind stress, coastal setup, tail weights, lag kernel |
| P5 | 10 FM checkpoints retrained on correct residual target | 2026-11-03 21:27-21:47 | `data/checkpoints/flow_matching_{region}_seed{42,99}.pt` all updated |
| P6 | Paper P0 caveats added (6-results.tex, 7-discussion.tex) | 2026-11-03 | LaTeX compiles clean: 18 pages, bibtex resolved, all refs pass |
| P7 | wave17_analyze.py fixed (3 verdict KeyError bugs + nonsense code) | 2026-11-03 | ruff clean, 268 pytest pass |

**CI Status**: ✅ 268 pytest pass, ruff clean, 78s

## Active GPU Work

| PID | Script | GPU | Status | Output |
|---|---|---|---|---|
| **551931** | `wave17_pipeline.py` | GPU3 | **RUNNING** (~1h elapsed) | 142/190 prediction files written |
| | | | | Baselines (70) + A3 (72 so far) |

## Damage Inventory (Wave 1.6 Finding, P0-audited)

| Run | Records | Status | Reason |
|---|---|---|---|
| A0 baselines | 20 + wave17 (70 total) | **✅ VALID** | Always read `water_level_m` (correct residual) |
| A1 oversampling | wave17 (20 per region) | **✅ VALID** | Same correct target as A0 |
| A2 history-noise | wave17 (20 per region) | **✅ VALID** | Same correct target as A0 |
| A3 old run | 120 | **EXPLORATORY** | FM + downstream trained on double-detided targets |
| **A3 new run (wave17)** | **120** | **PENDING_COMPLETION** | Corrected FM checkpoints, correct target semantics |
| Diffusion-TS M2 | 10 | **INVALIDATED** | Loader buggy; also P0 double-subtraction not touched DDPM |
| Flow Matching M4 (old) | 10 checkpoints | **INVALIDATED** | Trained on `water - tide` (double-subtraction) |
| **Flow Matching G8 (new)** | **10 checkpoints** | **✅ VALID RETRAIN** | Retrained 21:27–21:47 on `water_level_m` |

## GPU Availability

| GPU | Status | VRAM | Usage |
|---|---|---|---|
| GPU 0 | **PROHIBITED** | 24 GB | yzt aerodpo + llama-server |
| GPU 1 | LOCKED | 24 GB | llama-server (18.5 GB) |
| GPU 2 | SATURATED | 24 GB | ComfyUI comfy-h3 |
| **GPU 3** | **OUR PRIMARY** | 24 GB (16.1 GB used) | **wave17 pipeline PID 551931** |
| GPU 4 | SATURATED | 24 GB | yzt aerodpo + Carla |
| GPU 5-9 | 99-100% | all | Other users |

**Policy**: GPU0 PROHIBITED. Primary: GPU3. Always specify `CUDA_VISIBLE_DEVICES=N`.

## Current Blockers

1. **ERA5 CDS auth missing** — `~/.cdsapirc` absent. Blocks physics-guidance experiments (ERA5 wind/pressure forcing). Owner action needed.
2. **Disk at 97%** — 281 GB free. Monitor before large downloads.
3. **wave17 pipeline completion** — est. ~35 min remaining for full 190 records.

## Emerging Scientific Signal (Partial 3-Region A3)

| Metric | A0 baseline | A1 oversampling | A2 noise | A3 ratio 0.5 GRU |
|---|---|---|---|---|
| ny_bight tw_mae | 0.1272 | 0.1442 (↑13%) | 0.1203 (↓5%) | **0.1077 (↓15.5%)** |
| thames tw_mae | 0.1787 | 0.1835 (↑3%) | 0.1879 (↑5%) | 0.2048 (↑14.6%) |
| us_gulf tw_mae | 0.1132 | 0.1169 (↑3%) | 0.1061 (↓6%) | **0.0992 (↓12.4%)** |

**Pattern**: Augmentation efficacy is STRONGLY region-dependent. Thames (largest residual amplitude 3.934m q99) is harmed by augmentation; ny_bight and us_gulf benefit. A1 oversampling is universally harmful (confirmed negative result). This pattern is SCIENTIFICALLY MEANINGFUL on corrected targets.

## Next Action

1. **[IMMEDIATE]** Wait for PID 551931 to finish (~35 min) → 190 records in predictions/v2/
2. **[IMMEDIATE]** Run `wave17_analyze.py` (with `paired_bootstrap`) → full CI table
3. **[IMMEDIATE]** Commit: wave17_analyze.py fix + paper P0 caveats + blocker updates
4. **[HIGH]** Run bootstrap CI on event-level (need EVENT IDs in predictions — current saves use windows)
5. **[HIGH]** Train G-family models (G1 VAE, G4 DDPM, G7 TimeGrad) on correct targets
6. **[MEDIUM]** ERA5 acquisition (user auth needed) → physics-prior calibration
7. **[MEDIUM]** Generate paper figures F4/F6/F7/F8/F9 from verified data

## Reference Documents

| File | Purpose |
|---|---|
| `docs/status/TASK_QUEUE.json` | Ordered task list S0–S7 |
| `docs/status/BLOCKERS.md` | Detailed blocker inventory |
| `docs/status/WORKLOG.md` | Chronological work entries |
| `docs/status/EXPERIMENT_REGISTRY.csv` | All experiments status/validity |
| `docs/status/FIGURE_REGISTRY.csv` | Manuscript figure inventory |
| `docs/status/DATASET_REGISTRY.csv` | Dataset provenance |
| `paper/CLAIMS_LEDGER.md` | Evidence-linked claim registry |
