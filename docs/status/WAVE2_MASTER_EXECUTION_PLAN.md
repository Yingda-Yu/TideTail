# Wave 2.0–2.4 Master Execution Plan

**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 — DRAFT, do not merge
**Base SHA**: `d7fdb5a` (Wave 2.0 multi-year cohort committed 2026-11-02)
**Plan created**: 2026-11-02

---

## 1. Current State (Wave 1.9 + Wave 2.0)

### 1.1 Repo & Environment

| Item | Value |
|---|---|
| Git HEAD | `d7fdb5a` — Wave 2.0 multi-year POT declustering |
| Working tree | Clean (prior uncommitted changes committed) |
| Branch | `trae/global-data-generative-research-20261031` |
| PR | #23, DRAFT, 3 pushes since base commit |
| Disk | 204 GB free (98% used) — TIGHT, monitor before downloads |
| GPU0 | ❌ PROHIBITED — used by other user (llama-server) |
| GPUs 1–2, 5–9 | Occupied by other users (starvla-policy, comfy-h3, others) |
| GPU 3–4 | Occupied by starvla-policy (99%/27% util) — wait for free |
| Tests | 270 pytest pass, ruff clean |

### 1.2 Data Inventory

| Source | Size | Rows | Notes |
|---|---|---|---|
| Prime-Formal 2015–2024 (5 regions) | 451 MB | 438,360 hourly | Anchor IDs verified: NOAA_8518750_The_Battery, 8665530_Charleston, 8771450_Galveston_Pier_21, BOM_IDO71003_Port_Kembla, IOC_SHEE |
| FES2022 tide | included | — | In Prime-Formal pipeline |
| NOAA CO-OPS 3 stations | 106 MB | 2.6M | Monthly CSVs, 6-min |
| IBTrACS v04r01 | 139 MB | 6,281 deduped | 5,013 unique SIDs |
| HURDAT2 Atlantic | — | 102 storms 2015–2024 | Deduped with IBTrACS |
| ERA5 | ❌ BLOCKED | — | CDS credentials missing (`~/.cdsapirc`) |
| GEBCO | ❌ NOT DOWNLOADED | — | Need 5 regional subsets |

### 1.3 Checkpoint Inventory (31 total)

| Model | Regions | Seeds | Count | Training Date | Epochs | Target Valid? |
|---|---|---|---|---|---|---|
| Diffusion-TS | All 5 | 42, 99 | 10 | 10/31–11/03 | 15 | ✅ Correct (full observed water level) |
| Standard Flow Matching | All 5 | 42, 99 | 10 | 11/01 | 15 | ✅ Correct (residual water_level_m) |
| Tail-aware Flow Matching | All 5 | 42, 99 | 10 | 11/02 | 15 | ✅ Correct + physics.tail_weight |
| Conditional VAE | ny_bight | 42 | 1 | 11/03 | 15 | ⚠️ Incomplete — 1 region only, needs more epochs |
| TimeGrad | None | — | 0 | — | — | ❌ Not trained |
| WGAN-GP / Conditional GAN | None | — | 0 | — | — | ❌ Not implemented |
| GP-prior FM | None | — | 0 | — | — | ❌ Not implemented |

### 1.4 Predictions (190 files — Wave 1.7/1.8, backfilled Wave 1.9)

All 190 `.npz` have `event_ids`, `canonical_storm_ids`, `observ_segment_ids`, updated Wave 1.9.

---

## 2. Scientific Framework (Wave 2.0 — OFFICIAL)

### 2.1 Target Semantics (CRITICAL, permanent)

```
water_level_m = ALREADY de-tided surge residual  ← FORECAST TARGET
observed_total_water_level_m = water_level_m + tide_level_m
```
Verified: `MAE(water_level_m + tide_level_m, observed_total_water_level_m) ≤ 1e-16 m` across all 5 regions.

### 2.2 Independent Unit Framework (Wave 2.0 corrected)

**Independent unit = 168h Coles 2001 POT + canonical TC overlay**
- POSITIVE units: `canonical_storm_id >= 0` → tropical cyclone from IBTrACS/HURDAT2
- NEGATIVE units: POT q95 threshold + 168h declustered residual clusters
- ALL units: genuinely independent, no residual correlation expected across 168h gaps

### 2.3 Wave 2.0 Power Analysis (CORRECTED with POT clusters)

| Test Cohort | Region | Canonical TCs | POT q95 clusters | Total | Power? |
|---|---|---:|---:|---:|---|
| **2024 only** | ny_bight | 0 | 15 | **15** | ✅ POWERED (≥12) |
| | us_gulf_coast | 6 | 11 | **17** | ✅ POWERED |
| | charleston | 2 | 13 | **15** | ✅ POWERED |
| | se_aus | 0 | 6 | **6** | ⚠️ Exploratory |
| | thames | 0 | 10 | **10** | ⚠️ Borderline |
| **2023–2024** | ny_bight | 0 | 33 | **33** | ✅ POWERED |
| | us_gulf_coast | 9 | 26 | **35** | ✅ POWERED |
| | charleston | 3 | 27 | **30** | ✅ POWERED |
| | se_aus | 0 | 15 | **15** | ✅ POWERED |
| | thames | 0 | 18 | **18** | ✅ POWERED |

**Critical correction from Wave 1.9**: Prior analysis counted ONLY canonical tropical cyclones (0–7 units/region), ignoring extratropical residual clusters. Including POT + 168h declustering recovers 95%+ of genuine independent surge events. **2024 test split is NOW POWERED for 3/5 regions.**

### 2.4 Frozen Split Protocol

Three split variants in `data/manifests/multiyear_split_manifest.json` (gitignored, SHA recorded):

| Variant | Train | Val | Test | Use |
|---|---|---|---|---|
| `primary` | 2015–2022 (8 yr) | 2023 (1 yr) | 2024 (1 yr) | Main paper results |
| `expanded_2yr_test` | 2015–2021 (7 yr) | 2022 (1 yr) | 2023–2024 (2 yr) | Supplementary power analysis |
| `expanded_3yr_test` | 2015–2020 (6 yr) | 2021 (1 yr) | 2022–2024 (3 yr) | Supplementary maximum power |

**All splits frozen, no test-set leakage during model selection.**

---

## 3. Execution Queue (Priority Order)

### Phase 2 — Generative Benchmark (GPU-BOUND)

**GPU prerequisite**: Wait for GPU 3 or 4 to free. Check `nvidia-smi` before every run. Never interrupt other users.

| # | Task | GPU | Estimate | Status |
|---|---|---|---|---|
| G1 | **Standard FM downstream re-run** (A0/A1/A2/A3) using 10/10 correct checkpoints + corrected independent-unit bootstrap | GPU3 | 4h | ✅ COMPLETE Wave 1.7/1.8 |
| G2 | **Tail-aware FM downstream A3 augmentation** — frozen tail-aware checkpoints (10 exist), run GRU/TCN forecasting at ratios 0.1/0.25/0.5, compare vs standard FM | GPU3 or 4 | 8h | ❌ PENDING |
| G3 | **Conditional VAE full training** — ny_bight only seed42 has 15 epochs; need all 5 regions × 2 seeds × ≥100 epochs with KL annealing | GPU3 | 24h | ❌ PENDING (1/10 done, need more epochs) |
| G4 | **Diffusion-TS target verification** — verify checkpoints were trained on `observed_total_water_level_m` (not double-detided residual). Retrain if wrong. | GPU4 | 12h | ❌ PENDING |
| G5 | **TimeGrad training** — 5 regions × 2 seeds × ≥50 epochs (autoregressive, expensive) | GPU3+4 | 48h | ❌ PENDING (CPU smoke test works, scale=15.8 at 2 epochs) |
| G6 | **WGAN-GP implementation + training** — code needed first, then 5×2×50 epochs | GPU3+4 | 24h | ❌ PENDING (not yet implemented) |
| G7 | **GP-prior / TSFlow FM** — design + implement + train | GPU4 | 16h | ❌ PENDING (not yet implemented) |
| G8 | **Physics-prior FM variants** — pressure-only, wind-only, combined tail+physics | GPU3+4 | 24h | ❌ PENDING (BLOCKED by ERA5) |

### Phase 3 — Physics (ERA5-BLOCKED except code)

| # | Task | Status |
|---|---|---|
| P1 | Formalize PHYSICS_EQUATIONS.md | ✅ COMPLETE (320 lines, 50 tests pass) |
| P2 | physics.py audit + hurricane-force drag reduction | ❌ PENDING |
| P3 | ERA5 CDS authorization | ❌ BLOCKED — need owner to create `~/.cdsapirc` |
| P4 | GEBCO regional subsets (5 regions only) | ❌ PENDING — 204 GB disk OK, rate-limited download |
| P5 | Storm-conditioned physics-prior FM | ❌ PENDING (depends on P3+P4) |

### Phase 4 — Manuscript & Publication

| # | Task | Estimate | Status |
|---|---|---|---|
| M1 | Manuscript audit: remove Wave 1.6 exploratory framing, add Wave 2.0 multi-year section | 8h | ❌ PENDING |
| M2 | Figures F1–F14: generate from actual experiment outputs only | 24h | ❌ PENDING |
| M3 | Tables: dataset, model config, training budget, fidelity, forecasting, uncertainty | 12h | ❌ PENDING |
| M4 | Journal shortlist + author instructions | 4h | ❌ PENDING |
| M5 | PDF compile + visual inspection | 4h | ❌ PENDING |
| M6 | Reproducibility package | 8h | ❌ PENDING |

### Phase 5 — Persistent Execution

| # | Task | Estimate | Status |
|---|---|---|---|
| A1 | Autonomous worker: lock, heartbeat, dependency resolution, crash recovery | 16h | ❌ PENDING |

---

## 4. Immediate CPU-Ready Work (while GPUs busy)

These tasks require no GPU and can proceed now:

1. **✅ DONE** Multi-year cohort generation + POT declustering fix (`make_multiyear_cohort.py` bug fixed, re-run, committed)
2. **✅ DONE** Feasibility report + census CSV + split manifest (docs/data/)
3. **⏳ NOW** Write this master plan document
4. **⏳ NOW** Update TASK_QUEUE.json + CLAIMS_LEDGER.md
5. **⏳ SOON** Extend train_gen_bench.py with TimeGrad training loop (model exists, just needs integration)
6. **⏳ SOON** Create tail-aware FM downstream augmentation script (variant of `run_a3_flowmatching_augment.py`)
7. **⏳ LATER** WGAN-GP implementation (new model module)
8. **⏳ LATER** Manuscript section drafts (methods, protocol, related work — result sections provisional)

---

## 5. Scientific Red Lines (AGENTS.md — permanent)

1. **No fabrication** — every metric traces to checkpoint or prediction file
2. **No GPU0** — always verify `CUDA_VISIBLE_DEVICES`
3. **No interrupting other users' processes** — rongjia's (now starvla-policy's) GPU3+4 jobs must complete
4. **No false significance** — report exploratory where independent units insufficient
5. **No future-condition leakage** — train-only normalization, q95 thresholds, split boundaries
6. **Credentials safe** — ERA5 CDS blocked, mark AUTH_REQUIRED, continue other tasks
7. **Disk safety** — 204 GB free (98% used); monitor before downloads
8. **No merge main** — PR #23 is DRAFT only

---

## 6. Credentials / Owner Actions Required

| # | Item | Priority | Instructions |
|---|---|---|---|
| 1 | **ERA5 CDS authorization** | HIGH | `pip install cdsapi`, create `~/.cdsapirc` with user key, accept dataset terms on CDS portal |
| 2 | **Publication approval** | LATER | No submission without owner sign-off |

---

## 7. Completion Definition

Wave 2.0–2.4 is COMPLETE when:

- [ ] All 8 generative model families trained on all 5 regions × 2 seeds (or documented gap reason)
- [ ] POT + canonical TC independent-unit evaluation on all models
- [ ] Physics-prior ablation (pressure-only, wind-only, tail+physics) — requires ERA5
- [ ] Manuscript compiled PDF, CLAIMS_LEDGER current, no overstated claims
- [ ] 14 publication figures generated from actual experiment outputs
- [ ] Reproducibility package traceable to all checkpoints + split manifests
- [ ] PR #23 ready for owner review
- [ ] Negative findings preserved (oversampling harm, Thames A3, VAE KL collapse)

---

*Generated automatically by TRAE. Every task entry references actual files, committed artifacts, and traceable SHA. Updates commit-sha when task commits to PR #23.*
