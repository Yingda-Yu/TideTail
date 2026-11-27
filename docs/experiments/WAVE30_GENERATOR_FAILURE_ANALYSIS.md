# Wave 3.0/3.2 — Generator Failure Analysis (P2 70-cell refresh)

Companion to `WAVE30_GENERATOR_BENCHMARK.md` (accepted 70/70). Every failed
cell is listed with the criterion values that failed; nothing was re-tuned
after evaluation. Gate policy: `src/tidetail/quality.py` (hard scale window
[0.5, 2.0]; W1 ≤ 0.25; autocorr RMSE ≤ 0.15; q95 exceedance ratio ∈
[0.5, 2.5]; peak KS p ≥ 0.01; warnings ⇒ INCONCLUSIVE; any hard failure ⇒
REJECTED).

## 1. Diffusion-TS — generation failure (NaN pools)

- **Regions affected: NY Bight, US Gulf Coast, Southeast Australia** (9, 9
  and 8 strata NaN respectively); Charleston and Thames produce finite
  output.
- Finite pools are catastrophic anyway: background scale 13.4 (Charleston)
  and 5.7–24.2 (Thames); pooled extreme scale 3.9–24.2.
- Charleston/Thames finite cells collapse to **exact-duplicate ratio 0.998**
  — the same degenerate-output mode diagnosed in the Wave 2.7 recovery study
  (E22: terminal-step x̂₀ amplification, loss NaN by epoch 24). The refresh
  confirms the failure is architecture/implementation-level and
  region-independent, not a sampling-path artefact (the E26 sampler fix is
  in place here).
- Downstream consequence: its Wave 2.8 pilot NaN deltas are excluded from
  significance counting (`wave30_corrected_inference.json`), consistent with
  the 9/26 corrected harm count.

## 2. TimeGrad — dispersion blow-up

- All 10 cells REJECTED in all evaluated strata. Scale ratios range from
  8.3 (best regional extreme pool) to **1.2 × 10⁶** (background pool).
- Wasserstein-1 is correspondingly enormous where finite. The failure mode
  matches E22: the denoiser consumes autoregressive memory but raw-sum
  memory grows without bound across the 72-step rollout.
- Region spread is huge (background pool median 1.0e3, range to 1.2e6), so
  TimeGrad failures are not uniform — but no cell is even borderline.

## 3. DDPM — systematic over-dispersion and tail over-exceedance

- All 10 cells REJECTED. Background pools scale 3.1–6.9; extreme pools
  0.85–4.37. q95 exceedance ratios 3.8–13.8 (Thames worst: W1 1.4–2.98,
  q95r ≈ 12–13.8).
- Pattern is uniform across seeds (e.g. Charleston background scale 3.7–4.7
  across seeds 42/99) — a family-level bias, not seed noise.

## 4. CVAE — background over-dispersion (reversed vs. the Wave 1.x pilot)

- All 10 cells REJECTED, driven by background strata: scale 2.1–4.6,
  q95r 3.8–8.3, autocorr RMSE 0.15–0.77, peak KS p ≈ 0.
- Notably the direction reversed relative to the earlier under-dispersed
  pilot (0.44): at this budget the repaired decoder over-generates
  dispersion while its extreme pools are closer to nominal (median 1.40) —
  bulk and tail are again discordant, in the opposite direction from FM.

## 5. WGAN-GP — region-conditional collapse in Thames

- 4 regions INCONCLUSIVE (with seasonal-extreme hard failures at n ≤ 2 in
  Charleston/NY/Gulf and one US Gulf autumn-background rejection: scale
  2.114, q95r 4.600).
- **Thames/North Sea: REJECTED in every stratum for both seeds** — scale
  0.26–0.48, W1 0.28–0.53, q95r ≈ 0.000 (the critic mode-collapses onto
  near-constant output), peak KS p ≈ 0. This is the only family whose gate
  status is region-conditional in background strata, and it coincides with
  the region whose augmentation is most harmful downstream — an
  observation, not a causal claim (n = 1 region).

## 6. Flow Matching / tail-aware FM — passing background, failing extremes

- No background or moderate-tail stratum is REJECTED in any cell, except
  one tail-aware cell (US Gulf seed 99 autumn background: scale 2.002,
  q95r 4.898, peak KS p ≈ 0).
- Extreme pools (n = 4 events/region) are INCONCLUSIVE everywhere: median
  pooled scale 0.80 (range 0.40–1.84), i.e. **extreme-event dispersion is
  under-generated even while background dispersion is nearly exact**.
  Seasonal extreme cells with n ≥ 1 that were evaluated failed hard
  criteria (e.g. US Gulf summer extreme: scale 0.40, W1 0.29, autocorr 0.66,
  q95r 0.000).
- Consequence carried into the manuscript: FM's fidelity eligibility is a
  *background* statement; the paper's tail-fidelity statements remain
  INCONCLUSIVE and no downstream benefit is attributed to tail fidelity.

## 7. Cross-family observations

1. **Background–extreme discordance is the norm, not the exception.** Every
   family ranks differently across strata (CVAE: over-dispersed bulk /
   near-nominal tail; FM: exact bulk / under-dispersed tail). A single
   pooled fidelity score would have hidden every one of these patterns.
2. **Thames is the hardest region for every family** (worst W1 for DDPM,
   WGAN-GP collapse, CVAE q95r 7.6–7.7 in SE-Aus-like patterns absent but
   Thames DDPM q95r ≈ 12) — consistent with its 24.7 % NaN gaps, macrotidal
   regime and low signal-to-noise residual, and with its downstream
   augmentation harm (all four cells, +3.4…+4.3 %).
3. No NaN/Inf appeared in any evaluated metric outside the documented
   Diffusion-TS degenerate pools (validator + JSON scan: 234 NaN metric
   fields, all under `diffusion_ts` in the three affected regions).
4. No duplicate-cell, duplicate-key or checkpoint-integrity defect was found
   by the strict validator (70/70 OK).
