# Wave 2.8 — Seven-Family Generative Benchmark (PILOT)

**Date**: 2026-11-08
**Artifacts**:
- Merged: `data/results/wave28_gen_benchmark.json` (sha256 `3656e1ed706c1e5cc048be6e14719c67c8cfd2d227e866f935517ed4c5ae9170`)
- Preserved Charleston-only original: `data/results/wave28_gen_benchmark_charleston_defaults.json`
- Pre-merge NY-Bight-only backup: `data/results/backups/merge_20261108_wave28_gen_benchmark.json.bak`
- Runs: `data/results/runs/wave28bench_20261108T081124Z_e0cee1db/` (NY Bight, exit 0, 4562 s)
- Figure: `paper/figures/W28_benchmark.pdf`

**Scope**: `south_atlantic_charleston` + `ny_bight`, 7 generative families, matched
windows/epochs/seeds/downstream budget, CPU-only, zero hyperparameter search for every
family. Charleston executed 2026-11-08 under the pre-systemd launcher (preserved
verbatim); NY Bight executed 2026-11-08 under systemd unit `tidetail-wave28bench`
(run code SHA `1106ecbb`, identical protocol fields verified before merge: n_train,
n_test, n_eval_windows, epochs, batch_size, seed, downstream_seeds, audit_version,
fairness_contract).

## Interpretation contract (P0-4)

This benchmark is a **PILOT**, not a final model-family ranking:

- Every family ran **10 epochs with fixed defaults and zero search budget**.
  Equal epochs are **not** equal computational budgets and **not** evidence of
  adequate convergence.
- A family failing here failed *this implementation at this budget in this
  configuration*; the failure does not license the claim that the family is
  inherently inferior.
- Fidelity gate is the shared `src/tidetail/quality.py` policy: a **hard
  scale-ratio window [0.5, 2.0]** (outside ⇒ REJECTED); any other quality
  check (range coverage, Wasserstein-1, autocorrelation RMSE, peak KS, peak
  timing) raises a warning. **ACCEPTED** = all criteria satisfied;
  **INCONCLUSIVE** = no hard failure but unresolved warnings;
  **REJECTED** = ≥1 hard failure. INCONCLUSIVE is a necessary, not sufficient,
  condition; downstream deltas from REJECTED generators are not meaningful
  augmentations.
- Fidelity was evaluated on the **first 200 consecutive test windows** (early
  2024, single season — a sampling-bias caveat; see "Revised evaluation"
  below). All fidelity numbers inherit this caveat.
- Downstream inference: 7 families × 2 regions × 2 downstream proxies =
  28 comparisons, Bonferroni α = 0.05/28 = 0.0018; **11 comparisons survive —
  all harmful effects of REJECTED generators**; no beneficial effect and no
  fidelity-eligible (INCONCLUSIVE) generator's effect survives correction.
  Single generator seed (42) ⇒ no model-seed robustness claim is possible.

## Protocol (identical for both regions, verified before merge)

| Field | Value |
|---|---|
| n_train / n_test windows | 69,970 / 8,601 |
| n_eval_windows (fidelity) | 200 |
| epochs | 10 (fixed; no validation-based early stopping — documented limitation) |
| batch_size | 256 |
| generator seed | 42 |
| downstream seeds | [42, 99, 7] |
| target space | non-tidal residual for all families; Diffusion-TS trained on total water and reconstructed to residual via tide subtraction before scoring |
| downstream | augmentation_ratio 0.25; GRU + TCN proxies; extreme-unit MAE on event units; paired sign-flip on per-event unit errors |

## Per-family training budget (P0-4 disclosure)

### south_atlantic_charleston (A0 GRU extreme MAE 0.1058, 14 extreme units)

| Family | Params | Opt steps | Train s | Scale ratio | Fidelity | GRU Δ vs A0 | p (sign-flip) |
|---|---:|---:|---:|---:|---|---:|---:|
| flow_matching | 27,937 | 2,740 | 115.9 | 1.077 | INCONCLUSIVE | −0.68% | 0.257 |
| tail_aware_fm | 27,937 | 2,740 | 78.0 | 1.061 | INCONCLUSIVE | −0.57% | 0.333 |
| cvae | 63,273 | 2,740 | 365.7 | 2.775 | REJECTED | +2.06% | 0.028 |
| timegrad | 7,177 | 2,740 | 32.8 | 25,321.7 | REJECTED | +946.1% | 1.2e-4 |
| ddpm_cond | 27,937 | 2,740 | 92.6 | 3.870 | REJECTED | +3.81% | 4.9e-4 |
| wgan_gp | 55,586 | 2,740 | 354.8 | 0.959 | INCONCLUSIVE | +2.99% | 0.0028 |
| diffusion_ts | 39,425 | 2,740 | 123.2 | 29.462 | REJECTED | +88.3% | 1.2e-4 |

### ny_bight (A0 GRU extreme MAE 0.1173, 15 extreme units)

| Family | Params | Opt steps | Train s | Scale ratio | Fidelity | GRU Δ vs A0 | p (sign-flip) |
|---|---:|---:|---:|---:|---|---:|---:|
| flow_matching | 27,937 | 2,740 | 96.1 | 0.905 | INCONCLUSIVE | −1.61% | 0.023 |
| tail_aware_fm | 27,937 | 2,740 | 87.1 | 0.903 | INCONCLUSIVE | −1.65% | 0.019 |
| cvae | 63,273 | 2,740 | 158.9 | 2.164 | REJECTED | −2.38% | 0.039 |
| timegrad | 7,177 | 2,740 | 20.0 | 814,406.9 | REJECTED | +1057.6% | 6.1e-5 |
| ddpm_cond | 27,937 | 2,740 | 70.8 | 3.574 | REJECTED | +2.78% | 0.0026 |
| wgan_gp | 55,586 | 2,740 | 251.6 | 0.831 | INCONCLUSIVE | −1.37% | 0.280 |
| diffusion_ts | 39,425 | 2,740 | 97.1 | NaN | REJECTED | NaN | — |

## Findings (pilot-level, no family-ranking claims)

1. **Only the two FM variants and WGAN-GP are INCONCLUSIVE (not REJECTED) in
   both regions** under the shared hard scale gate [0.5, 2.0]: FM scale
   1.077/0.905, tail-aware FM 1.061/0.903, WGAN-GP 0.959/0.831. All
   INCONCLUSIVE cells still carry unresolved warnings (e.g. autocorrelation
   RMSE > 0.15 and peak-KS p < 0.01 for FM); none is ACCEPTED.
   (WGAN-GP 0.831 lies inside the hard gate but outside ±10% — it is not a
   ±10% pass.)
2. **CVAE, TimeGrad, DDPM and Diffusion-TS are REJECTED at this budget** —
   scale ratios 1.6× to 8×10⁵× outside [0.5, 2.0], and Diffusion-TS diverges
   to NaN in NY Bight (finite 29.5 in Charleston). These failures are
   statements about *these implementations under 10 epochs with zero search*,
   not about the families. They are consistent with the Wave 2.7 diagnostic
   pilots (CVAE posterior collapse; TimeGrad 6.5k-param denoiser
   under-capacity; DTS Fourier-repair divergence).
3. **Multiple-testing correction (corrected 2026-11-08): 11 of the 28
   downstream comparisons survive Bonferroni α = 0.0018 — and all 11 are
   HARMFUL effects of REJECTED generators**: TimeGrad (both regions × both
   proxies, +946% to +4×10⁵%), DDPM-cond (Charleston GRU/TCN +3.8%/+5.2%,
   NY Bight TCN +5.3%), Diffusion-TS (Charleston +88%/+106%; NY Bight NaN,
   p≈0 by degeneration). **No beneficial effect survives**, and **no
   fidelity-eligible (INCONCLUSIVE) generator shows a surviving effect**
   (best nominal: FM NY-Bight GRU −1.61% p=0.023; WGAN-GP Charleston
   +3.0% p=0.0028 — both above α). The FM/NY-Bight nominal improvement must
   additionally be read against the expanded-year protocol (Wave 2.8
   corrected rerun), where standard-FM augmentation *harmed* extreme-event
   MAE (+0.86%/+0.92%) under the leakage-controlled multi-year split.
   Rejected-generator deltas are statements about these implementations, not
   evidence that the model families are intrinsically ineffective.
4. **Training-time spread under equal epochs** (20 s … 366 s per family)
   confirms that "10 epochs" is not a common compute budget: TimeGrad
   consumed the least wall-time yet failed hardest; equal-epoch comparisons
   are budget-confounded.

## Fidelity-sample caveat (P3) and revised evaluation

The pilot evaluates fidelity on `test_res_t[:200]` — the **first 200
consecutive test windows** of the 2024 test split, i.e. a single continuous
early-winter slice (~11 days). Fidelity statistics therefore describe one
season and are **not** an annual-representative assessment. The pilot
artifact is preserved unchanged; a **revised, separately-labelled evaluation**
(season-stratified windows, independent extreme events, background periods,
Tail95/Tail99 per stratum, multiple generator seeds, validation-data selection
only) is queued to run after the physics experiment frees the CPU, and will be
reported as `wave28_gen_benchmark_revised.json` without overwriting the pilot.

## Provenance

- Generator seed 42; downstream seeds [42, 99, 7] (only GRU mean shown above;
  per-seed values in artifact); 14/15 event-level extreme units per region.
- Charleston region dict preserved verbatim from the earlier verified run;
  NY Bight dict from the systemd run; merge validated by field-by-field
  protocol equality (recorded above) — merge is 1:1, no result recomputation.
- All runs CPU-only (`CUDA_VISIBLE_DEVICES=""`).
