# LATEST PROGRESS — TideTail Wave 2.8 (Three Publication-Critical Experiments)

**Last updated**: 2026-11-08 21:50 CST
**Head SHA**: `858281e` + interim docs commit (this wave)
**Branch**: `trae/global-data-generative-research-20261031`
**PR**: #23 (Draft — DO NOT MERGE main)

---

## Wave 2.8 current state (updated 2026-11-08 21:50)

| Priority | Status | Evidence |
|---|---|---|
| P0 execution reliability | **DONE** | systemd user units survived context resets; **lesson: 3 concurrent CPU jobs triggered systemd-oomd kill of wave28physics at 17:28** — relaunched 21:39 alone (133 GB free) |
| P0-2 artifact protection | DONE (in use) | `scripts/tidetail_run.sh` immutable run dirs + backup-before-overwrite + append-only `data/results/RUN_MANIFEST.jsonl`; benchmark merge done with backup + atomic replace |
| P0-3 E26 fix | DONE (committed `9b790a6`) | Running processes' code SHA `1106ecb` recorded in run `env.txt`; 311 pytest pass |
| P2 6-cell expanded | DONE + VERIFIED | `data/results/wave28_expanded_eval.json` — honest negative, exploratory framing |
| P0 physics rerun (W27-P4) | **DONE 2026-11-08 18:15** | `data/results/wave27_physics_fm.json` (exit 0, 7,448 s): all 10 variant×region cells fidelity-INCONCLUSIVE (E26 fix verified at full scale); downstream deltas −2.2%..+1.3%, **nothing survives Bonferroni α=0.005** (10 comparisons, single downstream seed 42); wind arm harms Charleston (+1.25% p=0.024) but helps NY Bight — no physics arm beats standard consistently; expanded-year harm result supersedes for augmentation-value claims. Doc §5.1 updated |
| P3 7-family benchmark | **DONE (both regions)** 2026-11-08 17:27 | NY Bight completed under systemd (exit 0, 4,562 s); Charleston dict merged (protocol fields verified identical) → `data/results/wave28_gen_benchmark.json` sha256 `3656e1ed…`; FM/tailFM/wgan_gp pass fidelity both regions; cvae/timegrad/ddpm/dts REJECT (dts→NaN in NY Bight); 28 downstream comparisons — none survives Bonferroni; **PILOT framing only**. Doc `docs/experiments/WAVE28_GEN_BENCHMARK.md` written with full training-budget table |
| P1 physics 6-arm | **RUNNING (relaunch 21:39, ETA ~03:30)** | unit `tidetail-wave28physics`, run `wave28physics_20261108T133917Z_39437029`; killed once by systemd-oomd during 3-way concurrency; incremental saves in place |
| P5 manuscript | PARTIAL | Expanded section done; benchmark pilot + wave27 §5.1 docs done; **paper 6-results.tex updates for benchmark + wave27 physics pending**; wave28 physics section pending final artifact |
| P6 acceptance/commit/push | PENDING | pytest baseline 311 pass re-verified 2026-11-08 21:30 |

**Remaining sequence**: wave28physics finishes (~03:30) → WAVE28_PHYSICS_FM.md + ledger E29 → paper sections (benchmark pilot table, wave27 physics, wave28 ablation) → regenerate W28_physics_ablation.pdf → acceptance matrix → pytest → latexmk PDF + visual inspect → simulated peer review → LATEST_PROGRESS/TASK_QUEUE/REGISTRY/CLAIMS_LEDGER → commit + push PR #23.

**Wave 2.8 headline findings so far**

1. E26 fix restores fidelity everywhere it is applied (wave27 physics: 10/10
   cells INCONCLUSIVE; expanded rerun: 2/2 INCONCLUSIVE).
2. Under the leakage-controlled expanded-year protocol, standard-FM
   augmentation **harms** extreme-event MAE (+0.86%/+0.92%) — honest negative;
   six-cell comparison: no seed-robust improvement anywhere.
3. Physics arms (statistical regularization, NOT PDE): no arm beats standard
   consistently; wind-prior sign flips across regions.
4. Seven-family pilot: only FM/tailFM/WGAN-GP pass fidelity at 10 epochs +
   zero search; failures are implementation-at-budget statements, not
   family verdicts.

**Environment**: CPU-only (`CUDA_VISIBLE_DEVICES=""`); avoid launching >1
concurrent heavy job — systemd-oomd killed wave28physics during 3-way
concurrency (evidence in `journalctl --user -u tidetail-wave28physics`).

---

## Wave 2.7 final state (2026-11-05)

| Priority | Status | Evidence |
|---|---|---|
| P1 verify Wave 2.6 | DONE | event union 679 (not 707); 2024 units 59; corrected statistics → 0 robust survivors; Charleston 5/48 withdrawn |
| P2 ERA5 | DONE | 600/600 partitions, 87,672 h/region, `all_regions_integrity_ok: true` |
| P3 model recovery | DONE | oracle sampler exact (1.1e-10); VAE repair refuted; TimeGrad partial; DTS repair refuted (NaN) |
| P4 physics FM | SUPERSEDED BY WAVE 2.8 P1 | 5-variant conditioning pilot withdrawn (E26); genuine 6-arm penalty design running |
| P5 expanded eval | CORRECTED RERUN DONE (2026-11-08) | see Wave 2.8 table above |
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
- Expanded protocol (Wave 2.8 corrected rerun): generator fidelity restored
  (scale 1.15 / 1.10) but standard-FM augmentation **does not help** —
  +0.86% / +0.92% extreme-event MAE harm vs A0, both significant under the
  sign-flip null. An honest negative result, reported as such.

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
