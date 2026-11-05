# Wave 2.7 — Acceptance Record

**Date:** 2026-11-05
**Branch:** `trae/global-data-generative-research-20261031`
**Base commit:** `242093a`
**PR:** #23 — DRAFT, `main` not modified
**Environment:** GPU0 prohibited; all other GPUs occupied by other users at
launch. Every Wave 2.7 experiment ran on CPU with thread caps. No GPU job was
started, duplicated or terminated.

This file records what Wave 2.7 accepted, what it rejected, and what it
explicitly does **not** claim. It is the audit gate for the wave.

---

## P1 — Wave 2.6 verification

| Criterion | Result | Evidence |
|---|---|---|
| Event-union arithmetic correct | **PASS** — 679 (634 POT + 73 TC − 28) | `data/results/wave27_event_set_audit.json` |
| 2024 independent event count | **PASS** — 59 window-eligible units (15/15/14/6/9) | same |
| Charleston 5/48 findings confirmed | **FAIL — withdrawn** | `data/results/paired_significance_wave27.json` |
| GRU vs TCN difference characterised | **PASS** — GRU 86/160 (−0.17%), TCN 110/160 (−7.08%) | same |
| Actual event-level sample sizes | **PASS** — 6–15 units per region, 40/51 in expanded test | event audit + expanded artifact |
| No duplicated storms / window leakage | **PASS** — 0 duplicate timestamps; 8,601 test windows | `window_integrity` |
| 48 comparisons cover the search space | **FAIL — qualification recorded** | `docs/audits/WAVE26_PAIRED_SIGNIFICANCE_AUDIT.md` |
| No exploratory promoted to confirmatory | **PASS** — all Wave 2.6 significance language downgraded | audit docs + ledger |

Defects corrected in this wave:

1. Cross-architecture pairing (GRU candidate vs TCN baseline).
2. `paired_bootstrap` computed `p = 2(proportion+1)/(n_boot+1)`, inflating
   significance ~5,000×.
3. Monte-Carlo resolution floor `2/(n_boot+1) > α`, making survival impossible.

Regression tests: `tests/test_paired_stats.py`, `tests/test_event_set_arithmetic.py`.

**Accepted conclusion:** zero of 320 comparisons are seed-robust. Three
within-region survivors are seed-fragile (significant against one of two A0
seeds only). Baseline seed variance reaches 25%.

---

## P2 — ERA5

| Criterion | Result |
|---|---|
| All partitions present | **PASS** — 600/600 |
| No missing hours | **PASS** — 87,672/87,672 per region |
| Decode of wind and pressure | **PASS** — `u10`, `v10` (m s⁻¹), `msl` (Pa), `expver` 0001 |
| Timestamp coordinate | **PASS** — hourly, 119/119 monthly continuities |
| CRS / grid | **PASS** — 0.25° `regular_ll`, 17×17 |
| Units | **PASS** — SI, wind speed derived √(u²+v²) |
| Station alignment | **PASS** — one-to-one merge with row-count assertion |
| Five-region readiness matrix | **PASS** — `data/results/era5_readiness_matrix.json` |
| Anchor representativeness | **LIMIT** — 4.5–11.7 km from cell centre; not downscaled |

`all_regions_integrity_ok: true`. Physics experiments began immediately on the
two regions that were verified, without waiting for the others.

---

## P3 — Generative baseline recovery

| Model | Defect localised | Repair outcome | Accepted? |
|---|---|---|---|
| Conditional VAE | scalar rank-1 latent; no `nn.Module` | reconstruction 0.882→0.326, **scale 3.29→7.26** | **NO — refuted** |
| TimeGrad | AR memory ignored; unstable aggregation | **scale 10.55→6.09** (sum), mean diverges 1.22e5 | **NO — partial** |
| Diffusion-TS | Fourier term on the wrong target | **NaN by epoch 24** (÷√ᾱ ≈ 41× amplification) | **NO — refuted** |

Oracle-denoiser control: deterministic reverse pass recovers `x0` to
`1.1e-10` max absolute error, so the sampler is not the cause of any failure.

**Accepted conclusion:** no family is shown to be intrinsically ineffective;
two of three proposed repairs did not work and are preserved as negative
results. Scope: single region (Charleston), CPU, 20k training windows.

---

## P4 — Physics-guided Flow Matching

Status: **RUNNING** at the time of this record (5 variants × 2 regions, full-data
FM fit, CPU). Artifact `data/results/wave27_physics_fm.json` is written only on
completion; no physics number is claimed in this record.

Accepted design constraints (all verified before launch):

* Physics features are summarised **only** over the 96 h conditioning history;
  the 72 h horizon is never read (`tests/test_physics_causality.py`, 4 tests).
* Train-only normalisation statistics.
* Identical downstream budget (8,000 windows, 200 steps) for A0 and every arm.
* Explicit units: msl Pa, wind m s⁻¹, tendency Pa/24 h.
* Bathymetry variant **not** implemented — GEBCO not downloaded; must not be
  claimed.
* Labelled a retrospective **reanalysis-conditioned** experiment, not an
  operational forecast.

---

## P5 — Expanded 2015–2024 evaluation

| Criterion | Result |
|---|---|
| Train 2015–2020 / val 2021 / test 2022–2024 | **PASS** |
| 168 h embargo at both boundaries | **PASS** |
| No cross-split forecast-window leakage | **PASS** |
| Train-only preprocessing | **PASS** |
| Validation-only model selection | **PASS** (epoch 13 both regions) |
| Existing checkpoints not reused | **PASS** — retrained from scratch |
| Event-level paired uncertainty | **PASS** — 40 and 51 POT units |
| Multiple-comparison limitation reported | **PASS** — 2 tests, α=0.025 |
| Test of H1 | **FAIL — invalid, generators fail fidelity** |

Fidelity: scale ratio 3.97 / 4.76 against gate [0.5, 2.0] → **REJECTED**.
Downstream effect: +8.88% (CI [+0.0035, +0.0170] m) and +15.85% (CI
[+0.0183, +0.0262] m).

**Accepted conclusion:** the protocol and machinery are correct; the retrained
generators are under-fitted; an over-dispersed generator measurably harms the
forecaster. This is **not** evidence that good generative augmentation harms
forecasting, and these models must not be used to test H1.

---

## P6 — Manuscript and figures

| Item | Result |
|---|---|
| Title / abstract reframed | **PASS** |
| Introduction states H1 unchanged and unconfirmed | **PASS** |
| Verified event statistics included | **PASS** |
| Generator benchmark table from artifacts | **PASS** |
| Recovery diagnosis with refuted repairs | **PASS** |
| GRU/TCN and regional comparisons | **PASS** |
| Physics ablation section | **PENDING** (run in flight) |
| Figures generated only from artifacts | **PASS** — 6 figures |
| PDF compiled | **PASS** — 21 pages, no unresolved references |
| No invented results | **PASS** — placeholders marked explicitly |
| Figure registry updated | **PASS** — `paper/figures/` |

Figures: `W27_schematic`, `W27_event_census`, `W27_fidelity_gate`,
`W27_architecture_region`, `W27_recovery`, `W27_baseline_seed`.

---

## P7 — Continuous execution

* Task queue reconciled with real processes; three pilots ran concurrently
  (physics, expanded, recovery), and a duplicate expanded-eval process was
  detected and killed before it could overwrite the artifact.
* No ERA5 downloads were duplicated (dataset already complete).
* Progress report, experiment registry, figure registry and manuscript updated
  in the same wave.
* CPU pytest gate: **295 passed**.

---

## Explicit non-claims

1. No claim that generative augmentation improves extreme coastal forecasting.
2. No claim that it degrades forecasting in general (P5's harm is a
   bad-generator artefact).
3. No claim that VAE, TimeGrad or Diffusion-TS are intrinsically unsuitable.
4. No claim of operational forecasting: physics conditioning is
   reanalysis-based.
5. No claim of downscaled or harbour-scale accuracy from 0.25° ERA5.
6. No claim that the 320-test family exhausts the search space.

## Residual risks

* Physics pilot may finish with fidelity failures like P5; if so, it will be
  recorded as a readiness result, not a physics verdict.
* The expanded protocol needs fidelity-aware epoch selection before it can test
  H1.
* Thames' 24.7% NaN coverage limits any claim about that region.

## Next

1. Finish and record the physics pilot (P4).
2. Implement fidelity-aware FM selection, rerun the expanded comparison.
3. VAE free-bits / learned prior; stable Diffusion-TS parameterisation.
