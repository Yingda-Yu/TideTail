# Research Execution Plan — TideTail

**Created**: 2026-10-30
**Branch**: `trae/research-recovery-20261030`
**Base evidence**: PR #19 audit + PR #21 diffusion fix + EXECUTE mandate

---

## Phase P0 (COMPLETE — 2026-10-30)

- ✅ Repo audit, environment check, GPU status, data directory, checkpoint existence
- ✅ Branch inheritance analysis and reconciliation
- ✅ Merged PR #19 (data audit, 55 tests) + PR #21 (diffusion implementation, 19 tests) → **69 tests pass**
- ✅ Created audit docs (`TRAE_FULL_AUDIT.md`, `BRANCH_RECONCILIATION.md`)
- ✅ TeX Live minimal compile verified

## Phase P1 — Data Integrity & Formal Protocol (ACTIVE)

**Goal**: Freeze a scientifically signed-off data manifest. Cannot proceed to formal GPU training or held-out evaluation until G1 signoff.

### P1-A: Data Sources Audit
- [ ] Enumerate all six regions: source, license, station ID, vertical datum, raw time range
- [ ] Record unit transformation: observed_total_water → tidal_residual (deterministic FES2022 tide)
- [ ] Verify water_level_m column = observed_total_water - FES2022_tide (PR #20 report confirms this)
- [ ] Audit HK cross-cutting event `pearl_river_hong_kong_event_01149` ([78903, 78904] vs cut 78904)
- [ ] Map all 350 files to SHA256 → migration manifest → release inventory → region admission JSON

### P1-B: Canonical Independent Storm Grouping
- [ ] Implement proper meteorological storm declustering (not just threshold runs)
- [ ] Address PR #21's embargo_filter criticism (midpoint-only insufficient)
- [ ] Full 72h-history → 24h-forecast support interval embargo
- [ ] For hourly origin o: train block [a,b) permits only o ≥ a+72 AND o ≤ b-24
- [ ] HK boundary regression test

### P1-C: Train-Only Everything
- [ ] Scaler fit on train rows only (already done in pilot_data.py — verify across all regions)
- [ ] q95/q99 thresholds from train only (already in splits.py train_only_thresholds — verify usage)
- [ ] Interpolation fitting on train only
- [ ] Future covariate availability verification (ERA5 issue/forecast vintage)

### P1-D: Protocol Freeze
- [ ] Unified target: water_level_m (residual) vs observed_total_water? — DECIDE NOW
- [ ] Unified window contract: 72h→24h (pilot) vs 96h→72h (release) — DECIDE NOW
- [ ] All experiment configs must carry explicit target/horizon/lookback/split_id/event_id/station_id
- [ ] Freeze manifest JSON with hashes, counts, event IDs, station IDs
- [ ] Held-out perturbation invariance tests

**P1 Acceptance**: Frozen manifest, data card, independent storm counts, split stats, GO/NO-GO report.
**HUMAN SIGNOFF REQUIRED** on: license, datums, canonical storms, HK exclusion.

## Phase P2 — Reliable Forecasting Baselines

**Depends**: P1 frozen protocol

### P2-A: Metrics Implementation
- [ ] Per-metric hand-computable unit tests
- [ ] Correct denominators: precision/recall/false_alarm_rate/false_alarm_ratio
- [ ] Event-level bootstrap (paired by independent storm, not overlapping windows)
- [ ] Zero-event fold handling
- [ ] Per-lead-time, per-station, per-severity breakdown

### P2-B: Baseline Models
- [ ] Persistence (current as forecast, good reference)
- [ ] Seasonal naive (24h-lag seasonal — real implementation, not toy)
- [ ] GRU (matched capacity, training budget, seeds)
- [ ] TCN (same)
- [ ] Optional Transformer if budget allows

### P2-C: Augmentation Controls
- [ ] A1: True training oversampling (weighted loss or balanced sampler, not prediction noise)
- [ ] A2: Physics-justified perturbation (not unconstrained interpolation)
- [ ] A0–A2 on frozen train/val/test from P1

### P2-D: P2 Acceptance
- [ ] Baseline table with per-seed results
- [ ] Matched training budget evidence (steps, params, GPU time)
- [ ] Negative results preserved

## Phase P3 — DDPM Sampling Diagnosis & Fix

**Depends**: Can run concurrently with P2 (no GPU required for diagnosis)

### P3-A: Reproduce Historical Failures
- [ ] Verify checkpoint hash/size/path exists (prerequisite: re-verify SHA256)
- [ ] Run `scripts/diagnose_ddpm_cpu.py` → fresh diagnostics.json
- [ ] Compare: CUDA artifact vs CPU repro magnitude
- [ ] Re-confirm terminal alpha_bar ≈ 5.93e-7, amplification ≈ 1298×

### P3-B: Investigation
- [ ] Cosine schedule terminal sensitivity analysis (different beta schedules)
- [ ] Epsilon vs x0 vs v-prediction objective comparison
- [ ] Timestep weighting (importance sampling)
- [ ] Terminal step calibration with oracle clean sample
- [ ] Architecture width/depth sweep (current: 27K params, 4 blocks, width 32)
- [ ] Training budget analysis (historical: 3 epochs, investigate 3/10/30/60/100)

### P3-C: Critical Constraints
- [ ] Cannot just change inference schedule and call it fixed
- [ ] Cannot clip or post-hoc rescale as scientific recovery
- [ ] Must retrain after any schedule/objective change
- [ ] Must preserve original failure samples and checkpoint

### P3-D: Sampling Quality Gates
- [ ] Numerical finiteness
- [ ] Amplitude, SD, q95/q99 tail mass
- [ ] Spectrum and autocorrelation
- [ ] Extreme duration, climb/decay shape
- [ ] Nearest-neighbor memorization check
- [ ] Scale screen (already implemented — verify passing)

## Phase P4 — Controlled GPU Training

**Depends**: P1 frozen, P3 diagnosis complete, GPU availability verified per session

### P4-A: GPU Safety
- [ ] Pre-check nvidia-smi for free GPUs (0, 3, 4 are candidates)
- [ ] Verify no conflicting TideTail processes
- [ ] Set CUDA_VISIBLE_DEVICES explicitly
- [ ] OMP/MKL/OpenBLAS/NumExpr = 2 threads
- [ ] No auto-fallback to CPU

### P4-B: Training Budget Curve
- [ ] 3 epochs (historical pilot, known failure)
- [ ] 10, 30, 60, 100 epochs with matched batch
- [ ] Multi-seed stability (3 seeds minimum)
- [ ] Matched optimizer steps across conditions

### P4-C: Artifacts per Run
- [ ] Config with SHA, source/data hashes, seed, physical UUID
- [ ] Per-step epsilon MSE CSV
- [ ] Resource JSONL (GPU util, VRAM, temp, RSS)
- [ ] Checkpoint every 100 steps + final
- [ ] 8 generated samples + sensitivity + NN-RMSE diagnostics
- [ ] sample_scale.accepted field
- [ ] All results linkable to paper/CLAIMS_LEDGER.md

## Phase P5 — A0–A4 Augmentation Utility Evaluation

**Depends**: P2 baselines + P3/P4 fixed DDPM → only if generation quality gates pass

### P5-A: Fixed Budget
- [ ] Same downstream predictor for all A0–A4
- [ ] Same total optimizer steps
- [ ] Same tuning budget
- [ ] Same seeds

### P5-B: Main Comparisons
- [ ] A0: Real-only
- [ ] A1: Balanced sampler / duplication (not prediction noise)
- [ ] A2: Physically justified perturbation
- [ ] A3: Standard conditional diffusion
- [ ] A4: Tail-aware conditional diffusion
- [ ] A5: A4 + physical/spatial constraints (if feasible)

### P5-C: Held-Out Only
- [ ] Validation for hyperparameter tuning only
- [ ] Test = independent real storms, frozen before any runs
- [ ] Event-level paired bootstrap CIs
- [ ] q95/q99 tail, peak magnitude, peak timing, threshold detection

### P5-D: If No Gain
- [ ] Report negative result
- [ ] Analyze why (poor generation quality? insufficient diversity? bad predictor?)
- [ ] Move to methodology-focused paper

## Phase P6 — LaTeX Paper (PARALLEL from P0 end)

### P6-A: Paper Scaffold — NOW
- [ ] Create `paper/main.tex`, `paper/references.bib`, `paper/sections/`, `paper/figures/`, `paper/tables/`, `paper/build/`
- [ ] Choose LaTeX template (generic article for now, not CVPR/NeurIPS)
- [ ] LITERATURE_MATRIX.md with verified DOI/authors/years
- [ ] CLAIM_EVIDENCE_MATRIX.md template

### P6-B: Write with Evidence
- [ ] Introduction (extreme-event scarcity, forecasting challenge, falsifiable H1/H2)
- [ ] Related Work (coastal forecasting, data augmentation, diffusion, storm surge)
- [ ] Data Protocol (P1 evidence — whatever is verified)
- [ ] Method (P3 diagnosis and fix — whatever is verified)
- [ ] Experiments (placeholder tables, explicitly marked as not run)
- [ ] Negative Results section (historical DDPM failure — already has evidence)
- [ ] Discussion, Limitations, Conclusion

### P6-C: Compile & Verify
- [ ] latexmk -pdf -halt-on-error
- [ ] PDF exists, no LaTeX errors
- [ ] No undefined references
- [ ] Figures/tables within bounds

## Delivery Order (Non-Blocking Parallelism Allowed)

```
P0 ✅  ←────────────── COMPLETE
P1   ├── P2 ──┐
    ├── P3 ──┤  (P1,P2,P3 independent)
    └── P6 ──┤  (paper parallel throughout)
              ↓
             P4 ──→ P5 ──→ P6 (complete)
```

## Immediate Next Steps

1. **P1-B**: Fix embargo_filter (full support interval, not just midpoint)
2. **P1-D**: Decide unified target (residual vs total water) and window contract (72h→24h vs 96h→72h)
3. **P3-A**: Verify checkpoint SHA256, re-run CPU diagnosis
4. **P6-A**: Create paper scaffold, first LaTeX compile with evidence-based content
5. Commit P0 audit + merge on `trae/research-recovery-20261030`
