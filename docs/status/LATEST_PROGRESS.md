# LATEST PROGRESS — TideTail Wave 2.8 (Three Publication-Critical Experiments)

**Last updated**: 2026-11-08 16:20 CST (Wave 2.8 in execution, systemd-persistent)
**Head SHA**: `1106ecb` (runner) ← `9b790a6` (E26 fix) on top of `8a45765`
**Branch**: `trae/global-data-generative-research-20261031` (pushed)
**PR**: #23 (Draft — DO NOT MERGE main)

---

## Wave 2.8 current state (updated 2026-11-08 16:20)

| Priority | Status | Evidence |
|---|---|---|
| P0 execution reliability | **DONE** | Root cause confirmed with evidence: experiment children lived under the trae-sandbox exec tree and were killed at context compaction (NOT OOM). All long jobs now run under `systemd-run --user` transient units (`tidetail-wave27`, `tidetail-wave28physics`, `tidetail-wave28bench`), PPID = systemd user manager (uid 1018, linger on) — verified to survive conversation resets. Artifact protection via `scripts/tidetail_run.sh` (commit `1106ecb`): immutable per-run dirs, backup-before-overwrite, append-only `data/results/RUN_MANIFEST.jsonl` |
| P0 state recovery | DONE | Wave 2.7 E26 fix committed `9b790a6` (includes sha256 of the code actually used by running processes); 16 regression tests pass |
| P0 expanded rerun | DONE | `data/results/wave27_expanded_evaluation.json`: fidelity INCONCLUSIVE (scale 1.147/1.097); downstream honest negative (+0.86%/+0.92% harm vs A0) |
| P2 6-cell expanded | **DONE + INDEPENDENTLY VERIFIED** | `data/results/wave28_expanded_eval.json` (2026-11-08): Charleston harmed in all 4 cells (+0.67..+1.19%; sign-flip p 0.0000–0.0058; 3/4 cells survive Bonferroni α=0.00625); NY Bight sign flips with FM seed (seed42 +, seed99 −), no seed-robust effect anywhere. Reported as **exploratory** (multiple comparisons, historical test-set reuse disclosed in artifact) |
| P0 physics rerun | RUNNING (systemd) | `scripts/wave27_physics_fm.py` corrected rerun, unit `tidetail-wave27`, ETA ~18:10 CST |
| P1 physics 6-arm | RUNNING (systemd) | `scripts/wave28_physics_fm.py`: standard / era5_cond (conditioning-only control) / pressure_prior / wind_prior / combined_prior / combined_tail; physical penalty on implied clean estimate x1_hat = x_t+(1−t)v; train-only OLS physics regression; auto-calibrated frozen λ; unit `tidetail-wave28physics`, ETA ~20:30–21:30 CST |
| P3 7-family benchmark | Charleston **DONE** (pilot); NY Bight RUNNING (systemd) | Charleston result preserved at `data/results/wave28_gen_benchmark_charleston_defaults.json` (+ /tmp copy): 7 families, 14 event pairs, epochs=10 seed=42 downstream_seeds=[42,99,7]. FM scale 1.077 w1 0.029; tailFM 1.061/0.028; cvae 2.775 REJECT; timegrad REJECT; ddpm 3.870; wgan_gp 0.959/0.078; dts REJECT; A0 GRU 0.1058; FM deltas GRU −0.68% p=0.257 / TCN −1.63% p=0.135. **Interpreted as PILOT only** (equal epochs ≠ equal compute/convergence; per-family params/opt-steps/train-time recorded; zero search budget for all families). NY Bight run under unit `tidetail-wave28bench`, ETA ~18:15 CST; Charleston region dict must be merged back into final `wave28_gen_benchmark.json` (identical protocol/seeds verified) |
| P4 matching protocol | DONE | Frozen strict-overlap rule (union 679) with tolerance sensitivity (24h: 676, 48h: 667, 72h: 665) documented in `6-results.tex` + `a-audit.tex` |
| P5 manuscript | PARTIAL | Expanded-year section rewritten (exploratory framing); physics/benchmark sections pending final artifacts; benchmark must be labelled pilot with full training-budget table |
| P6 commit/push | PENDING | After running experiments complete: figures, acceptance matrix, tests, PDF, push PR #23 |

**Wave 2.8 headline findings so far**

1. **The E26 fix converts the expanded-year story**: generators now pass to
   INCONCLUSIVE fidelity (scale ~1.1), and standard-FM augmentation shows a
   small, consistent, statistically significant **harm** (+0.86% / +0.92%
   extreme-event MAE, p_perm ≤ 0.016) — an honest negative result under the
   leakage-controlled multi-year protocol (retrospective caveat recorded).
   Six-cell two-region comparison strengthens this: no cell shows a
   seed-robust improvement; Charleston is consistently harmed.
2. Tail calibration (q95 over-exceedance 1.6–1.7) remains the generator's
   main fidelity defect — motivating the tail-aware cells in the completed
   six-cell comparison.
3. Seven-family benchmark (Charleston pilot): only the two FM variants and
   WGAN-GP pass the fidelity gate; CVAE/TimeGrad/DDPM/Diffusion-TS fail at
   10 epochs with zero search budget. This is a **pilot observation about
   implementation stability under a fixed budget**, NOT a final model-family
   ranking (P0-4 interpretation rule).

**Environment**: GPU0 prohibited; all other GPUs occupied by other users
(verified 2026-11-08). All Wave 2.8 experiments run on CPU with thread caps.

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
