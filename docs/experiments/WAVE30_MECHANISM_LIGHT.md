# WAVE30 P4-light — Regional Heterogeneity, Severity and Season Dependence

**Script**: `scripts/wave30_mechanism_light.py`
**Output**: `data/results/wave30_mechanism_light.json`
**Run**: 2026-11-26, RNG seed 20261126, B_boot=10,000, B_perm=5,000
**Inputs**: `wave30_expanded_event_metrics.csv` (1,440 rows = 180 POT events ×
4 arms × 2 fm seeds, 100% joined to `multiyear_declustering.json`), P2
fidelity job untouched (analysis is single-core, ~2 min).

## Status

DONE for the three data-only mechanism probes. The training-based P4 arms
(pressure/wind priors, ERA5-only control, combined physical prior) remain
queued behind P2 — serial heavy jobs.

## Disclosure

All three analyses are descriptive mechanism probes computed on the
retrospective test set. Severity (POT-cluster `peak_residual_m`, and
`peak/threshold` excess ratio as sensitivity) and season labels come from
the declustering manifest whose thresholds are train-referenced; season
labels use local meteorological seasons with southern-hemisphere mapping
for `southeast_australia` (DJF = austral summer). Nothing here feeds model
selection.

## 1. Regional heterogeneity — CONFIRMED, not merely descriptive

Cochran-style Q with cluster-bootstrap SEs (df = 4), per augmentation arm:

| Arm | Q | p (χ², df=4) | Pairwise contrasts surviving Holm |
|---|---|---|---|
| standard_gru | 51.9 | 1.4e-10 | 8/10 |
| standard_tcn | 78.3 | 4.0e-16 | 8/10 |
| tail_aware_gru | 53.4 | 7.2e-11 | 8/10 |
| tail_aware_tcn | 76.2 | 1.1e-15 | 6/10 |

Per-region mean per-unit effect (standard_gru; %, cluster-bootstrap 95% CI):

| Region | Δ% vs A0 | CI95 | n events |
|---|---|---|---|
| ny_bight | −0.05 | [−0.45, +0.33] | 51 |
| south_atlantic_charleston | +1.34 | [+0.60, +2.14] | 40 |
| southeast_australia | −1.28 | [−2.04, −0.68] | 20 |
| thames_north_sea | +5.95 | [+3.74, +8.33] | 32 |
| us_gulf_coast | +0.33 | [−0.17, +0.86] | 37 |

Interpretation: region identity is a **statistically confirmed** moderator
of the augmentation effect (all four arms, p ≪ 0.0001). The paper may now
state regional heterogeneity as a tested finding, not just a descriptive
pattern; the architecture-level claim (TCN vs GRU) remains limited to the
paired interaction results in `wave30_architecture_interaction.json`
(survivors in beneficial regions only).

## 2. Event-severity dependence — effects attenuate at high severity

Spearman ρ between per-event mean Δ% and event peak residual, permutation
p, Holm across the 20 tests (5 regions × 4 arms):

| Region | std_gru ρ | std_tcn ρ | tail_gru ρ | tail_tcn ρ | verdict |
|---|---|---|---|---|---|
| southeast_australia | **+0.74** | +0.62 | **+0.74** | +0.60 | benefit shrinks as severity rises |
| thames_north_sea | **−0.68** | **−0.53** | **−0.68** | **−0.54** | harm shrinks as severity rises |
| south_atlantic_charleston | **−0.46** | −0.34 | **−0.46** | −0.34 | harm shrinks as severity rises (GRU sig.) |
| us_gulf_coast | −0.11 | −0.02 | −0.11 | +0.02 | null |
| ny_bight | −0.01 | −0.07 | −0.01 | −0.07 | null |

Bold = p_holm < 0.05. Median-split check (standard_gru): SE-Aus
low-severity −2.14% vs high-severity −0.42%; Thames +10.09% vs +1.81%;
Charleston +2.52% vs +0.17%.

**Mechanism statement**: augmentation effects — beneficial AND harmful —
are concentrated in low-to-moderate severity events and attenuate for
extreme events in every region where a severity gradient exists. This is
consistent with generative augmentation mostly informing the model about
frequent-regime structure rather than rare-regime dynamics, and it
motivates the P3 tail-calibration arms.

## 3. Seasonal dependence — NULL

Across-season permutation tests (F-style statistic, Holm within arm across
regions): **no region × arm survives Holm** (raw p 0.036–0.95; smallest
Holm-adjusted p > 0.15, Gulf-Coast TCN arms). Season is NOT a supported
moderator at these sample sizes (2–15 events per season cell; e.g. SE-Aus
MAM n=2, Thames SON n=2). Power caveat recorded; no seasonal claim is
made.

## Conclusion for the paper

1. Regional heterogeneity: tested and confirmed (new result, upgrades the
   descriptive framing in §4b of the master status).
2. Severity: augmentation acts mainly on low/moderate events — supports
   tail-calibration motivation (P3) without any test-set selection.
3. Season: no evidence of seasonal moderation — do not claim one.

Negative/null results (Gulf/NY severity, all season tests) are preserved
as honest findings.
