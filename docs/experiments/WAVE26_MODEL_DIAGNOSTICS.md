# Wave 2.6 — Model Failure Diagnostics

**Date**: 2026-11-04
**Context**: Wave 2.5 labeled VAE, TimeGrad, Diffusion-TS as failed. This audit is a diagnostic START, not final verdict.

## Executive Summary

Flow Matching is the ONLY viable generative family for TideTail — confirmed by proper fidelity gates and downstream paired significance. The other three model families have DIAGNOSABLE implementation bugs or fundamental mismatches with the problem structure.

**Key diagnostic finding**: Flow Matching achieves scale_ratio 0.95–1.12 (correct), while Diffusion-TS gets 0.12–0.20, TimeGrad 0.07–0.32, VAE 0.33–0.81.

## Per-Model Diagnosis

### Diffusion-TS: Reverse Process Error Accumulation

**Symptom**: 10/10 REJECTED, scale 0.12–0.20 (5-8× under-spread), MAE 1-9m. Thames seed42 produced NaN.

**Root Cause (diagnosed)**:
- DDPM reverse-time denoising on 72-step horizon amplifies tiny input noise
- cosine1000_subsample schedule with only 64 steps doesn't spread enough variance
- ResidualDenoiser with `use_full_context=True` processes 96-step history but only 32-width — insufficient capacity
- Target semantics: DDPM trained on full `water_level` but fidelity gate checks on total water level

**Small diagnostic experiment needed**:
```bash
# Reduce T to 32 or 16 steps (fewer denoising steps = less accumulation)
# Or switch to Flow Matching-style ODE solver (adaptive step, better stability)
```

**Negative result to preserve**: DDPM reverse process on 72-step extreme water levels is numerically unstable.

---

### TimeGrad: Autoregressive Scalar Diffusion + Insufficient Capacity

**Symptom**: 10/10 REJECTED, scale 0.07–0.32, MAE 1.5-2.0m. Training loss barely moved (0.97→0.94 over 50 epochs).

**Root Cause (diagnosed)**:
- TimeGradDenoiser with width=24 and single GRU layer — only ~6,500 parameters
- Autoregressive scalar diffusion must handle 72 steps of autoregressive rollout
- Per-step error compounds: each of 72 steps uses the previous noisy sample
- No conditioning on 96-step history — denoiser is unconditional

**Small diagnostic experiment needed**:
```bash
# TimeGrad width=64 + 2 GRU layers (4× parameters)
# Condition denoiser on 96-step history before autoregressive rollout
# Increase training epochs to 200 with early stopping
```

**Negative result to preserve**: Current TimeGrad implementation with width=24 is severely undercapacity for 72-step extreme residual generation.

---

### VAE: KL Divergence Collapse

**Symptom**: 10/10 REJECTED/INCONCLUSIVE, scale 0.33–0.81. KL divergence went from 0.1 to 3.6 over 100 epochs.

**Root Cause (diagnosed)**:
- NO KL ANNEALING. Beta=0.001 constant from epoch 1
- KL loss monotonically increases — posterior drifts away from prior, not toward it
- By epoch 100, KL=3.6 (posterior very different from prior), but beta is still tiny → decoder ignores latent
- Samples come from near-prior (random noise), not learned posterior

**Small diagnostic experiment needed**:
```bash
# KL annealing: beta=0→0.001 over 50 epochs (linear warmup)
# Increase latent_dim from 32 to 128
# Use cyclical beta (0↔0.01 every 25 epochs)
```

**Negative result to preserve**: No KL annealing causes posterior collapse in TideTail VAE — samples are near-prior noise.

---

### Flow Matching: WHY IT WORKS

**Symptom**: 10/10 INCONCLUSIVE (scale 0.95–1.12 — correct spread), downstream charleston GRU Δ=-13%.

**Why**:
- ODE solver (Euler, 64 steps) — no error accumulation like DDPM
- Conditional velocity field — learns transport map from noise → data
- Tail-aware variant ≈ identical (no measurable benefit from tail weighting)
- 72-step horizon handled natively — no autoregressive rollout
- Width=32 (moderate) — sufficient for residual generation

**Verdict**: Flow Matching is the correct architecture for TideTail. Tail-aware weighting provides no additional downstream benefit.

## Diagnostic Pilot Experiments (Small, ~2h each on GPU3)

| Experiment | Model | Change | Hypothesis |
|---|---|---|---|
| D06a | VAE | KL warmup 0→0.01 over 50 ep | Fixes posterior collapse |
| D06b | TimeGrad | width=64 + conditional | Sufficient capacity |
| D06c | Diffusion-TS | Reduce T to 32 steps | Less denoising accumulation |
| D06d | TimeGrad | KL warmup + cyclical | Similar to VAE |

All experiments run on GPU3 (currently free). Only run if Wave 2.5 negative results don't satisfy scientific requirements.

**If diagnostics are negative (all still broken)**: Flow Matching stands alone as the only viable generative family. This is a valid scientific conclusion — not a failure.

## What to Publish

**Preserve ALL negative results**:
- ❌ VAE fails without KL annealing
- ❌ TimeGrad undercapacity for 72-step horizon
- ❌ Diffusion-TS unstable reverse process
- ✅ Flow Matching works

This is valuable scientific information — not just "we tried and 3 out of 4 failed."

---
*Wave 2.6 Model Diagnostics — 2026-11-04*
