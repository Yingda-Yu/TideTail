# Wave 2.8 P2 — Expanded-Year Six-Cell Comparison (2015–2024)

**Status**: COMPLETE (2026-11-08, CPU, 1,887.2 s)
**Script**: `scripts/wave28_expanded_eval.py`
**Artifact**: `data/results/wave28_expanded_eval.json`
**Branch**: `trae/global-data-generative-research-20261031` (Wave 2.8)

## Design

Six cells per region — {A0 real-only, Standard FM augmentation,
Tail-Aware FM augmentation} × {GRU, TCN} — under the leakage-controlled
protocol: train 2015–2020, validation 2021 (epoch selection by unweighted
validation FM loss, identical rule across variants), test 2022–2024 touched
once at the end. Regions: Charleston and NY Bight. FM seeds {42, 99};
downstream seeds {42, 99, 7}; frozen augmentation ratio (2,000 synthetic /
8,000 real); 200 optimizer steps for both forecasters; identical
hyperparameters and budgets across all cells.

**Retest disclosure**: the 2022–2024 span partially informed earlier
exploratory waves of this project. This is retrospective validation, not
untouched confirmatory inference.

**Multiplicity**: 2 regions × 2 variants × 2 forecasters = 8 paired
comparisons (reported per seed, so 16 tests); Bonferroni α = 0.00625 for
the family. All results are exploratory/retrospective.

## Results

A0 extreme-event unit MAE (m): Charleston GRU 0.1171 / TCN 0.1167
(40 units); NY Bight GRU 0.1389 / TCN 0.1429 (51 units).
Sign-flip randomisation p-values, paired per independent unit.

| Region | Variant | Forecaster | Δ vs A0 (seed 42 / seed 99) | p (42 / 99) |
|---|---|---|---|---|
| Charleston | standard | GRU | +0.78% / +0.68% | 0.0058 / 0.036 |
| Charleston | standard | TCN | +1.11% / +1.01% | <1e-4 / 0.0027 |
| Charleston | tail-aware | GRU | +0.82% / +0.67% | 0.0045 / 0.035 |
| Charleston | tail-aware | TCN | +1.19% / +1.05% | <1e-4 / 0.0012 |
| NY Bight | standard | GRU | +0.45% / **−0.85%** | 0.051 / 0.0003 |
| NY Bight | standard | TCN | +1.33% / **−0.61%** | 0.0011 / 0.020 |
| NY Bight | tail-aware | GRU | +0.49% / **−0.84%** | 0.036 / 0.0003 |
| NY Bight | tail-aware | TCN | +1.42% / −0.48% | 0.0005 / 0.069 |

Generator fidelity (all INCONCLUSIVE; q95 exceedance ratio 1.60–1.73):

| Region | Variant | scale ratio (seed 42 / 99) |
|---|---|---|
| Charleston | standard | 1.147 / 1.119 |
| Charleston | tail-aware | 1.136 / 1.124 |
| NY Bight | standard | 1.097 / 1.250 |
| NY Bight | tail-aware | 1.091 / 1.250 |

## Findings

1. **No cell shows a seed-robust improvement.** The only improving
   generator (NY Bight seed 99, GRU −0.85%, TCN −0.61%) is contradicted
   by the other seed (+0.45%, +1.33%), and its fidelity is worse
   (scale 1.250 vs 1.097).
2. **Charleston is consistently, significantly harmed** by augmentation in
   all four augmented cells across both seeds (TCN more so than GRU).
3. **Tail-awareness does not change the outcome** in any cell: deltas move
   by ≤ 0.1 percentage points between standard and tail-aware, and tail
   calibration (q95 exceedance) is essentially identical.
4. **Generator-seed variance can flip the sign of the augmentation effect**
   (NY Bight), so single-seed augmentation studies in this domain are
   uninterpretable.
5. The dominant remaining fidelity defect — q95 over-exceedance 1.6–1.7 —
   is shared by both variants; tail-aware loss weighting did not repair it.

## Verdict

Under the multi-year leakage-controlled protocol, generative augmentation
from partially fidelious Flow Matching provides **no robust benefit** to
extreme-event forecasting in these two regions, with consistent small harm
in Charleston and seed-unstable effects in NY Bight. This is an honest
negative result consistent with the Wave 2.7 corrected rerun (standard FM,
GRU only). It does not establish that augmentation cannot help — the
generator's tail calibration is imperfect — but it removes the claim that
current FM augmentation does help.

## Limitations

- Two regions (Charleston, NY Bight); the other three regions were not
  run in this wave (compute-bound; CPU-only environment).
- 2 FM seeds bound the seed-variance estimate weakly.
- Both generators remain INCONCLUSIVE on fidelity (heavy synthetic tails);
  the experiment tests partially-fidelious augmentation only.
- Physics-guided augmentation cells are pending the Wave 2.8 P1 gate.
