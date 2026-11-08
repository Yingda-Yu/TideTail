# Wave 2.6 — Paired Significance Audit

**Date**: 2026-11-04
**Motivation**: Wave 2.0 bootstrap compared candidate CI against A0 POINT ESTIMATE — not a proper paired test.

## Executive Summary

Charleston **does have statistically significant FM augmentation benefit** — but only for the GRU forecaster, only at specific augmentation ratios, and only after proper paired bootstrap with multiplicity correction. Five configs survive Bonferroni (α=0.00104). The TCN forecaster shows NO benefit.

## What Wave 2.0 Bootstrap Did Wrong

The previous bootstrap checked: `candidate_ci_high < A0_point_estimate`. This is **not a significance test** — it compares one group's variability to another group's point estimate. A correct paired test must:

1. Compute **per-event paired differences** (A3 loss - A0 loss for each independent event)
2. Bootstrap the **distribution of paired differences** 
3. Compute a **p-value** for the null hypothesis that the mean difference = 0

## Method (This Audit)

**Paired Bootstrap**: 10,000 resamples of the 15 independent units. Per-unit MAE computed, then differences bootstrapped.

**Paired Randomization Test**: 10,000 permutations of A0/A3 labels within each pair. Exact p-value under H0 of exchangeability.

**Multiplicity**: Bonferroni correction across all 48 paired comparisons (2 A0 seeds × 24 A3 configs). α_per_comparison = 0.05/48 = 0.00104.

**Data**: 15 independent units from `south_atlantic_charleston_*.npz` event_ids — same event_ids for every prediction file (assigned by wave20 bootstrap).

## Key Results

### Survive Bonferroni (p < 0.00104) — 5 configs

| A3 Config | vs A0 | Δ% | CI95 | p_boot | p_perm |
|---|---|---:|---|---:|---:|
| `A3_fm_s42_ratio_0.1_gru_seed99` | A0_s42 | **-15.6%** | [-2.6%, -0.7%] | 0.0000 | 0.0048 |
| `A3_fm_s99_ratio_0.1_gru_seed99` | A0_s42 | **-16.0%** | [-2.5%, -0.9%] | 0.0000 | 0.0006 |
| `A3_fm_s42_ratio_0.25_gru_seed42` | A0_s42 | **-12.6%** | [-2.0%, -0.6%] | 0.0002 | 0.0056 |
| `A3_fm_s42_ratio_0.1_gru_seed99` | A0_s99 | **-11.5%** | [-1.8%, -0.5%] | 0.0000 | 0.0070 |
| `A3_fm_s99_ratio_0.1_gru_seed99` | A0_s99 | **-11.9%** | [-1.9%, -0.6%] | 0.0000 | 0.0016 |

### Verdict by Forecaster

| Forecaster | Configs Tested | Bonferroni Sig | Not Sig | WORSE |
|---|---:|---:|---:|---:|
| GRU | 24 | **5** | 14 | 5 |
| TCN | 24 | 0 | 10 | **14** |

**Scientific interpretation**: Flow Matching augmentation helps GRU by approximately 10-16% but does NOT help TCN. This is a forecaster-family effect, not universal.

### Best Augmentation Ratio

| Ratio | GRU Delta Range | Bonferroni Hits |
|---|---|---:|
| 0.1 (10% synthetic) | -4% to -16% | **4/5** |
| 0.25 (25% synthetic) | -1% to -13% | **1/5** |
| 0.5 (50% synthetic) | -1% to -9% | **0/5** |

**Less is better** — augmentation ratio 10% consistently outperforms 25% or 50%.

### FM Model Seed Irrelevance

FM seed 42 vs 99 produces similar results. The augmentation ratio matters more than the FM random seed.

## Limitations and Disclosures

### Cross-Seed Comparison
Some tests compared A3_fm_s42_gru_seed99 vs A0_real_only_gru_seed42 (different GRU seeds). The correct comparison would be A3 vs A0 with the **same GRU seed**. The 5 surviving configs include 3 cross-seed and 2 same-seed comparisons. This is a **post-hoc exploratory finding** — not pre-specified.

### Selection Bias
Augmentation ratios (0.1, 0.25, 0.5) were chosen after seeing initial results. This inflates false-positive risk via multiple testing. We corrected with Bonferroni but acknowledge this is a conservative remedy.

### Single Region
Only charleston has enough units (n=15) for meaningful significance. se_aus has n=6 (UNDERPOWERED), thames has n=9 (BORDERLINE). Generalizing "FM augmentation works" to all regions requires expanded_3yr_test retraining.

### GRU-only Finding
Significance is GRU-specific. Do not claim "FM augmentation improves TideTail forecasting" universally — the claim must be qualified: **"FM augmentation improves GRU residual MAE at charleston, but the effect does not generalize to TCN."**

## What Wave 2.5 Overstated

| Claim | Wave 2.5 | Wave 2.6 Corrected |
|---|---|---|
| "Charleston statistically significant" | ✅ Confirmed | ✅ Confirmed — but for GRU only, not universal |
| "4 configs verified" | ✅ Confirmed | ✅ 5 configs survive (actually 5, not 4) |
| "Survives Bonferroni" | ✅ Claimed | ✅ Verified with p_boot < 0.001 |
| "General improvement across forecasters" | ⚠️ Implied | ❌ **TCN is WORSE** — forecaster-specific effect |
| "General improvement across ratios" | ❌ Not claimed | ❌ Ratio 0.5 shows no benefit |
| "Generalizable to other regions" | ⚠️ Suggested | ❌ **Not demonstrated** — charleston only |

## Confirmatory vs Exploratory

| Category | Verdict |
|---|---|
| **Confirmatory** (pre-specified tests) | None — no pre-registration exists |
| **Exploratory — Charleston GRU benefit** | ✅ Strong (5/48 Bonferroni, both bootstrap and randomization tests agree) |
| **Exploratory — TCN benefit** | ❌ Negative (14/24 worse) |
| **Exploratory — Ratio effect** | ⚠️ Moderate (0.1 > 0.25 > 0.5) |
| **Exploratory — Cross-region** | ❌ Not powered enough |

## Audit Recommendation

1. **Downgrade Wave 2.5's "universal significance" claim** → frame as "charleston GRU exploratory finding"
2. **Expanded_3yr_test retraining** is needed for cross-region generalization
3. **Report TCN negative result honestly** — do not sweep under the rug
4. **Ratio 0.1 only** for future physics experiments — ratios 0.25/0.5 provide diminishing returns

---
*Wave 2.6 Paired Significance Audit — corrected 2026-11-04*
*Script: scripts/paired_significance_test.py → data/results/paired_significance_wave26.json*

## WAVE 2.7 INDEPENDENT RERUN (2026-11-05) — SUPERSEDES ALL FINDINGS ABOVE

The Wave 2.6 result ("Charleston 5/48 Bonferroni, GRU-only") does **not** reproduce.
Three defects were found and fixed:

1. **Cross-architecture pairing.** The Wave 2.6 script paired TCN candidates
   against GRU A0 baselines (and vice versa). The corrected script pairs a
   candidate only with A0 runs of the *same* forecaster.
2. **Bootstrap p-value overflow.** The p-value applied a count-style `+1`
   correction to a *proportion*, shrinking p by ~5,000×. A comparison with an
   observed effect of −0.235 and CI95 [−0.70, +0.20] was reported as p=0.00023.
   With the fix, that case returns p≈0.03 and `ci_excludes_zero=False`.
3. **Monte-Carlo resolution.** With 10,000 resamples the smallest attainable
   two-sided p is 2/(n+1)=2.0e-4, which is *above* the family threshold
   1.56e-4 — survival was mathematically impossible. Resampling is now
   vectorised and raised to 30,000 (floors 6.7e-5 / 3.3e-5).

Corrected specification: 5 regions × 2 architectures × 16 candidates × 2 A0
seeds = **320 comparisons**, Bonferroni α = **1.563e-4**, extreme independent
units only (`event_id != -999`).

| Result | Wave 2.6 | Wave 2.7 (corrected) |
|---|---|---|
| Charleston Bonferroni survivors | 5/48 | **0** |
| Robust survivors across the 320 family | — | **0** |
| Nominal survivors (ignoring baseline seed) | — | 3, all TCN, each vs ONE A0 seed only |
| GRU directional effect | "benefits" | 86/160 improved, mean Δ **−0.17%** |
| TCN directional effect | "worse" | 110/160 improved, mean Δ **−7.08%** |

The three nominal survivors (ny_bight TCN ratio 0.25 seed42 and ratio 0.1 s99
vs A0 seed 42; southeast_australia TCN ratio 0.5 seed99 vs A0 seed 99) each
survive against only one of the two A0 baseline seeds. Zero candidates survive
against both, so these are **baseline-seed artifacts**: A0 TCN MAE differs by
up to 25% between seeds (se_aus 0.0713 vs 0.0569), which is larger than the
claimed augmentation effect.

Per-region mean Δ (candidate minus A0, negative = improvement):

| Region | GRU | TCN |
|---|---:|---:|
| ny_bight | −3.41% | −13.14% |
| us_gulf_coast | −2.33% | −5.37% |
| south_atlantic_charleston | −4.30% | −0.09% |
| southeast_australia | −0.59% | −21.69% |
| thames_north_sea | **+9.78%** | **+4.88%** |

Thames/North Sea worsens under augmentation for both architectures. The
Wave 2.6 story ("GRU benefits, TCN does not") is reversed: on average TCN
improves more, but the effect does not survive family-wide correction and is
not robust to baseline seed. **Treat as exploratory and unresolved.**

### Search-space coverage (qualification)

The 320-comparison family covers every candidate *prediction file* that exists
in `data/predictions/v2` (A1 oversampling ×2 seeds, A2 history noise ×2 seeds,
A3 FM ×2 generator seeds × ratios 0.1/0.25/0.5). It does **not** enumerate the
full historical exploratory space: tail-aware A3 predictions are absent, and
discarded ratios, generator hyperparameters, epoch budgets and model-selection
choices were never materialised as predictions. Bonferroni over 320 is therefore
**anti-conservative** relative to the true search. No confirmatory claim follows.

### Verdict

> No augmentation configuration shows a family-corrected, baseline-seed-robust
> improvement on extreme independent units in any region. Charleston's Wave 2.6
> "5/48" finding is withdrawn. Exploratory directional evidence suggests TCN may
> benefit more than GRU on average, but region and seed variability dominate the
> effect. Confirmatory tests require the expanded 2015–2020 train / 2022–2024
> test protocol with pre-registered candidates.

Output: `data/results/paired_significance_wave27.json`; regression tests in
`tests/test_paired_stats.py`.
