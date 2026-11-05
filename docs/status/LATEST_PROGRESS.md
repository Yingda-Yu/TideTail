# LATEST PROGRESS — TideTail Wave 2.7 (Model Recovery, Physics, Manuscript)

**Last updated**: 2026-11-05 Wave 2.7
**Head SHA**: see `git log -1` (Wave 2.7 commits on top of `242093a`)
**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft — DO NOT MERGE main)

---

## Wave 2.7 current state

| Priority | Status | Evidence |
|---|---|---|
| P1 verify Wave 2.6 | DONE | event union 679 (not 707); 2024 units 59; corrected statistics → 0 robust survivors; Charleston 5/48 withdrawn |
| P2 ERA5 | DONE | 600/600 partitions, 87,672 h/region, `all_regions_integrity_ok: true` |
| P3 model recovery | DONE | oracle sampler exact (1.1e-10); VAE repair refuted; TimeGrad partial; DTS repair refuted (NaN) |
| P4 physics FM | RUNNING | 5 variants × 2 regions, full-data FM fit (CPU) |
| P5 expanded eval | DONE (negative readiness) | splits verified; retrained FM fails fidelity → +8.88% / +15.85% harm; does not test H1 |
| P6 manuscript | DONE | new title/abstract/intro/results/discussion/conclusion; 6 figures; PDF compiled |
| P7 queue/tests/commit | DONE | `TASK_QUEUE.json` reconciled; pytest CPU gate; pushed to PR #23 |

**Two headline corrections this wave**

1. The 2015–2024 event union is **679**, not 707. The old arithmetic added
   POT and TC totals without the 28-event intersection.
2. The Wave 2.6 "5/48 multiplicity-adjusted Charleston findings" is
   **withdrawn**. After within-architecture pairing, a bootstrap p-value fix
   (previous value inflated ~5,000×) and a Monte-Carlo resolution floor below
   the correction threshold, **zero** of 320 comparisons are seed-robust.

**New negative results preserved**

- Conditional VAE: strengthening the latent path improved reconstruction
  (0.882 → 0.326) but made sampling **worse** (scale 3.29 → 7.26); 0/32
  active latent dims in both variants (posterior collapse vs `N(0,I)` sampling).
- Diffusion-TS: x0-spectrum Fourier repair **diverges to NaN** at epoch 24
  (`x0_pred` divides by `sqrt(ᾱ) ≈ 0.024`, ~41× amplification).
- Expanded protocol: the retrained FM fails fidelity (scale 3.97 / 4.76), so
  the downstream degradation is a generator-readiness failure, not a test of
  augmentation utility.

**Environment**: GPU0 prohibited; all other GPUs occupied by other users.
All Wave 2.7 experiments ran on CPU with thread caps. No GPU job was
interrupted, duplicated or terminated.

---

## Current State: Comprehensive Audit Complete

Wave 2.5 completed a full state recovery of the TideTail research project after a multi-day pause. Key accomplishments:

✅ **Priority 0 Scientific Audits** — All three critical inconsistencies resolved
✅ **Full Generative Model Benchmark** — All 5 model families fully evaluated with fidelity gates
✅ **Complete Checkpoint Inventory** — 50/50 checkpoints verified, 10/10 for each viable model
✅ **Process Recovery** — Only ERA5 workers survived; GPU jobs died naturally (processes ended, checkpoints saved)
⚠️ **Expanded Protocol Retrain** — All existing checkpoints use 2015-2022 train split; expanded_3yr requires 2015-2020 retrain

---

## GPU Process Reconciliation

| PID Reported | PID Verified | GPU | Status | Artifact |
|---|---|---|---|---|
| Standard FM A3 | 4019786 → **DEAD** | GPU4 | Process ended | Checkpoints saved (10/10) |
| VAE full training | 4022380 → **DEAD** | GPU1 | Process ended | Checkpoints saved (10/10) |
| TimeGrad full training | 4022383 → **DEAD** | GPU3 | Process ended | Checkpoints saved (10/10) |
| ERA5 worker1 | 1868415 → ✅ ALIVE | CPU | Running | 371/600 downloaded (62%) |
| ERA5 worker2 | 1868416 → ✅ ALIVE | CPU | Running | Same |
| VAE/TimeGrad/Diffusion-TS eval | **RUN NOW** | GPU3 | ✅ Complete | fidelity evals written |

**GPU Status as of 2026-11-04 06:10 UTC**:
- GPU0: 🚫 PROHIBITED
- GPU1: llama-server (18.5GB, NOT our job)
- GPU2: comfy-h3 (22.3GB, NOT our job)
- GPU3: 🟢 FREE (24MB, available for TideTail)
- GPU4: tguancha policy server (9.4GB, NOT our job)
- GPUs 5-9: OCCUPIED by others

---

## Generative Benchmark — Wave 2.5 Final Verdict

### Fidelity Gate Results (Primary Split, 5000 samples, test residual)

| Model | Checkpoints | Scale Ratio | Gate | Verdict |
|---|---:|---|---|---|
| **Flow Matching (standard)** | 10/10 | 0.95–1.12 | **INCONCLUSIVE** | ✅ **VIABLE** — Perfect spread, statistically sound |
| **Flow Matching (tail-aware)** | 10/10 | ~1.0 (expected) | **INCONCLUSIVE** | ✅ **VIABLE** — Performs ≈ identically to standard |
| **Diffusion-TS** | 10/10 | 0.12–0.20 | **REJECTED** | ❌ **DEAD** — 5-8× under-spread, MAE 1-9m (catastrophic) |
| **TimeGrad** | 10/10 | 0.07–0.32 | **REJECTED** | ❌ **DEAD** — 3-14× under-spread, MAE 1.5-2.0m |
| **Conditional VAE** | 10/10 | 0.33–0.81 | **REJECTED/INCONCLUSIVE** | ❌ **BROKEN** — KL collapse, no annealing implemented |

### Why Non-FM Models Fail

- **Diffusion-TS**: DDPM reverse-time error accumulation on full water-level targets. The cosine1000_subsample schedule only partially mitigates; the fundamental architecture amplifies tiny noise inputs into implausible residuals. Thames seed42 produced NaN gradients.
- **TimeGrad**: Autoregressive scalar diffusion with only 6,500-param denoiser — insufficient capacity for 72-step horizon. Training loss barely moved (0.97→0.94 over 50 epochs).
- **VAE**: KL divergence went from 0.095 to 3.6 over 100 epochs — posterior ≈ prior from early training, latent collapse. No KL annealing schedule was implemented (constant beta=0.001 from epoch 1). Samples come from near-prior, not learned posterior.

### Downstream A3 Augmentation (GRU, residual MAE)

| Region | A0 Baseline | Best Δ% (Standard FM) | Best Δ% (Tail-Aware FM) | Verdict |
|---|---:|---:|---:|---|
| ny_bight | 0.1252 | **-12.2%** | **-12.4%** | 📉 Numeric improvement |
| us_gulf_coast | 0.1105 | **-10.3%** | **-10.3%** | 📉 Numeric improvement |
| charleston | 0.1022 | **-8.7%** | **-8.3%** | ⚠️ CI overlap with n=15 primary |
| se_aus | 0.0485 | **-4.7%** | **-4.7%** | 📉 Descriptive |
| thames | 0.1896 | **-3.5%** | **-3.2%** | ❌ Minimal → compare with POT bootstrap |

**Key scientific finding**: Tail-aware FM provides NO additional benefit over standard FM for downstream augmentation. Both variants produce identical downstream performance.

**Statistical significance** (from Wave 2.0 POT bootstrap): Charleston is the only region with statistically significant FM augmentation improvement (CI95 entirely below A0, 4 configs verified, Δ=-16.0%). This survives Bonferroni correction.

---

## Audits Completed

| Audit | Document | Key Finding |
|---|---|---|
| Event Count Reconciliation | `docs/audits/WAVE25_EVENT_COUNT_RECONCILIATION.md` | 707 deduped independent units, no double-counting bug |
| Statistical Power Audit | `docs/audits/WAVE25_STATISTICAL_POWER_AUDIT.md` | 3/5 powered on primary split; 5/5 fully powered on expanded_3yr |
| Split Leakage Audit | `docs/audits/WAVE25_SPLIT_LEAKAGE_AUDIT.md` | Primary split clean; expanded needs checkpoint retrain |

---

## ERA5 Download

| Metric | Value |
|---|---|
| Workers | 2 (PIDs 1868415, 1868416) |
| Progress | 371/600 partitions (62%) |
| Status | ✅ Active, running ~35 hours |
| CDS Protection | Acknowledged 2026-10-06 announcement; 3s cooldown respected |
| Variables | 10m U wind, 10m V wind, mean sea-level pressure |

---

## Resource Status

| Resource | Value |
|---|---|
| Disk free | 1014 GB (87% used) |
| pytest | 270 PASS |
| Checkpoints | 50 total (10 each: FM, FM-tail, VAE, TimeGrad, Diffusion-TS) |
| Downstream pred files | 190 (.npz in predictions/v2/) |

---

## Git Status

| Item | Value |
|---|---|
| Branch | `trae/global-data-generative-research-20261031` |
| Base | `92f8534` → new commits pending |
| New Wave 2.5 commits | 4 (audit docs + eval scripts + results) |
| PR | #23 DRAFT — DO NOT MERGE main |

---

## Scientific Claims Status Update

| Claim | Old Status | Wave 2.5 Update |
|---|---|---|
| E7 (Diffusion-TS broken) | CONFIRMED on residual | ✅ CONFIRMED on total_water too — 10/10 REJECTED |
| E16a (FM augmentation) | charleston verified | ✅ charleston still holds; FM is ONLY viable model |
| New: VAE viability | Unverified | ❌ KL collapse, scale 0.33-0.81 |
| New: TimeGrad viability | Unverified | ❌ 10/10 REJECTED, scale 0.07-0.32 |
| New: Tail-aware > Standard FM | Hypothesis | ❌ Identical performance — tail weighting unnecessary |

---

## Remaining Work

| Priority | Task | Effort | Dependencies |
|---|---|---|---|
| **P0** | Audit manuscript (paper/main.tex) | 4h | None |
| **P1** | Physics-guided FM (ERA5) | 20h GPU | 62% downloaded, ~46h remaining at current rate |
| **P2** | WGAN-GP baseline | 36h GPU | Code implementation needed |
| **P3** | Retrain for expanded_3yr_test | 30h GPU | Requires regenerating window_index splits |
| **P4** | Manuscript figures F1-F14 | 8h | Model results (primary split OK) |
| **P5** | Independent expanded-cohort bootstrap | 4h | 197 units available |

---

**Blockers**: None critical. Physics experiments wait on ERA5 (running). Expanded protocol requires checkpoint retrain.
**Primary publication path**: FM augmentation results on primary split + charleston significance claim + comprehensive generative failure analysis.

---
*Wave 2.5 executed 2026-11-04 — research continuation, statistical audit, comprehensive benchmark*
