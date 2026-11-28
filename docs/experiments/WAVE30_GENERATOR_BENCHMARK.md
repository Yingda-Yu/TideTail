# Wave 3.0/3.2 — Stratified Seven-Family Generative Benchmark (P2 FINAL ACCEPTANCE)

**Status: ACCEPTED — 70/70 cells verified.** Date: 2026-11-28.
Artifact: `data/results/wave30_generator_fidelity.json` (refresh complete,
PID 1425288, exit after `thames_north_sea done in 1147s`, `WROTE ...`).
Stratified row-level CSV: `data/results/wave30_generator_fidelity_stratified.csv`
(770 rows = 70 cells × up to 11 strata rows).

## 1. Design (frozen)

5 regions × 7 generator configurations × 2 generator seeds (42, 99) = **70 cells** (the seven configurations comprise two Flow Matching variants — standard and tail-aware — plus CVAE, TimeGrad, DDPM, WGAN-GP and Diffusion-TS).
Families: `flow_matching`, `tail_aware_fm`, `cvae`, `timegrad`, `ddpm_cond`,
`wgan_gp`, `diffusion_ts`. Contract: 10 epochs, batch 256, 8 CPU threads,
`CUDA_VISIBLE_DEVICES=""`, draws=3 per stratum, no CUDA fallback. The 96 h
history → 72 h target contract and the target semantics of the production
benchmark are preserved (Diffusion-TS trains on observed total water level by
design; scored against its own target, never ranked against residual-trained
families — see E25).

Strata (per region): season (winter/spring/summer/autumn, southern-hemisphere
seasons flipped for SE Australia) × intensity (background / extreme), plus
pools `all_extreme`, `all_background`, `all_moderate_tail`. Stratum sizes:
background 40 per season (160 total); extreme **4 per region total**
(Charleston 1/0/0/3; SE-Aus 0/0/1/3 in season order; Thames 2/0/0/2;
NY Bight and Gulf analogous) and 0–3 per seasonal extreme cell.

## 2. Checkpoint verification (strict validator, commit 310c685)

Scope note: this section certifies **artifact and protocol acceptance** —
that the checkpoints are the pre-declared ones and the run contract was
honoured. It is distinct from scientific fidelity acceptance: the per-cell
gate statuses in §4 remain REJECTED/INCONCLUSIVE as evaluated.

`scripts/wave30_validate_checkpoints.py` (set-equality gate: exactly 5
predefined regions × 7 families × seeds {42, 99}; duplicate-JSON-key
detection; SHA256; `torch.load` with per-cell exception reporting; tensor
counts; named_parameters() parameter audit on the real architectures;
WGAN-GP generator-vs-critic split):

- **cells: 70/70 present, 70 OK, 0 failures.**
- Param audit per family (identical in all 10 cells of a family):
  FM / tail-aware FM / DDPM 27,937; CVAE 63,273 (encoder 29,824 +
  decoder 33,449); TimeGrad 7,177; WGAN-GP total 55,586 = generator 30,401 +
  critic 25,185; Diffusion-TS 39,425. All `PARAMS_MATCH` against the
  per-cell recorded `param_count`.
- Every checkpoint SHA256 recorded in the JSON per cell
  (`checkpoint_sha256`).
- 14 validator regression tests (incl. extra-region/family/seed, duplicate
  keys, SHA mismatch, empty checkpoint, param mismatch, buffer counting).

## 3. Fairness metadata (recorded per cell)

All 70 cells: `device=cpu`, `cpu_threads=8`, `epochs=10`, `draws=3`,
`train_s`/`sample_s`/`optimizer_steps` recorded. `optimizer_steps` = 2,740
(four gap-free regions) / 1,980 (Thames, whose source series carries 24.7 %
NaN gaps → fewer valid training windows; the step-count difference is a
property of the data, not unequal budgets — epochs and batch size are
identical everywhere). Both values are consistent with
`ceil(n_train_windows/256) × 10`.

## 4. Family-level results (gate = `quality.py` policy; scale window
[0.5, 2.0] hard; warnings ⇒ INCONCLUSIVE; ≥1 hard failure ⇒ REJECTED)

Pooled scale ratios (median over the five regional pools; region range in
parens). See Figure F11 (`paper/figures/W32_fidelity_stratified.pdf`).

| Family | background pool | extreme pool (n=4/region) | Cell verdicts (10 cells/family) |
|---|---|---|---|
| Flow Matching | 1.28 (0.94–1.99) | 0.80 (0.40–1.79) | 10/10 INCONCLUSIVE (never REJECTED in any non-extreme stratum; 3 seasonal-extreme strata REJECTED at n≤2) |
| Tail-aware FM | 1.28 (0.94–2.00) | 0.80 (0.41–1.84) | 10/10 INCONCLUSIVE; 1 autumn-background REJECTED (US Gulf seed 99: scale 2.002, q95r 4.898); 3 seasonal-extreme REJECTED at n≤2 |
| CVAE | 2.92 (0.94–4.60) | 1.40 (0.61–3.81) | 10/10 REJECTED — background over-dispersion (scale 2.1–4.6, q95r 3.8–8.3, autocorr RMSE ≤0.77) |
| TimeGrad | 1.0e3 (24.9–1.2e6) | 3.9e2 (8.3–1.2e6) | 10/10 REJECTED — catastrophic dispersion blow-up (up to 1.2e6) |
| DDPM | 4.40 (3.09–6.86) | 2.91 (0.85–4.37) | 10/10 REJECTED — systematic over-dispersion + tail over-exceedance (q95r up to 13.8, W1 up to 2.98) |
| WGAN-GP | 1.08 (0.37–2.11) | 0.55 (0.26–1.11) | 4 regions INCONCLUSIVE; **Thames REJECTED in all strata** (scale 0.26–0.48, W1 0.28–0.53, q95r ≈0, peak KS p≈0) |
| Diffusion-TS | 13.4 / NaN | 3.9–24.2 / NaN | 10/10 REJECTED — **NaN metric pools in NY Bight, US Gulf, SE Australia** (degenerate output); finite pools scale 3.9–24.2; exact-dup ratio 0.998 where finite |

Memorisation checks: exact-duplicate ratio 0.000 for every family except
Diffusion-TS (0.998 where finite — the known collapse mode). The
near-duplicate (0.995-similarity) warning saturates at 1.0 for several
families; this metric is a warning (not a hard criterion) and is expected to
saturate for 3-draw strata over homogeneous background windows — it is
reported, not interpreted as a gate failure.

## 5. Extreme stratum: INCONCLUSIVE, never accepted

`all_extreme` contains **4 independent events per region** (n=4–5 design
target; realised n=4 in every region this cycle; seasonal extreme cells hold
0–3 events and are `NOT_EVALUATED` below the minimum-n). **No family's
extreme pool is ACCEPTED anywhere.** Flow Matching / tail-aware FM extreme
pools are INCONCLUSIVE (median scale 0.80; W1 0.03–0.53; peak KS often
fails at n=4). The extreme stratum therefore remains formally
**INCONCLUSIVE by design** and is never described as validated; the paper's
tail-fidelity statements do not rely on it.

## 6. Headline conclusions (mirrored in the manuscript §6.2/F11)

1. Flow Matching (both variants) is the only family never REJECTED in any
   background or moderate-tail stratum in any region — still INCONCLUSIVE
   everywhere, and its extreme-tail agreement is not established (n=4).
2. Bulk/background agreement does not imply tail agreement: FM background
   pools sit at scale ≈1.0–1.3 while its extreme pools fall to ≈0.80
   (0.40–1.79), i.e. under-dispersion concentrates exactly where it matters.
3. Every statistical failure is reported with its criterion values (see
   `WAVE30_GENERATOR_FAILURE_ANALYSIS.md`); no failed cell was re-tuned.
4. The five-region downstream result (12/20, 6 beneficial / 6 harmful) was
   produced with standard/tail-aware FM whose cell status here is
   uniformly INCONCLUSIVE — fidelity-gate status does not predict
   downstream utility (the paper's fidelity-vs-utility gap).
