# Wave 3.2 — Final Statistical Audit

**Date**: 2026-11-27
**Scope**: independent verification of every inferential claim that will
appear in the manuscript, before the Conclusion is frozen.
**HEAD at audit**: b61d09c (arch-interaction correction) on
`trae/global-data-generative-research-20261031`.
**Machinery audited**: `scripts/wave28_physics_fm.py::paired_event_stats`
(shared by all experiments), `scripts/wave30_five_region_aggregate.py`,
`scripts/wave30_arch_interaction.py` (v2),
`scripts/wave30_mechanism_light.py`.

---

## 1. Inference machinery (one implementation, audited once)

All paired tests in the project call `paired_event_stats`:

- **Test**: two-sided sign-flip randomisation of per-event-unit paired
  deltas about zero (exchangeability under H0: zero mean effect; the
  bootstrap CI is reported for effect size only and is *never* used as
  the null).
- **Exact vs Monte Carlo**: exact enumeration for n ≤ 16 units
  (2^n sign patterns, includes the identity flip, so p ≥ 2^-n, no zero
  p possible); Monte Carlo (n_boot = 10,000, seed 0, deterministic)
  otherwise.
- **MC resolution floor**: p_min = (b+1)/(n_boot+1) = 1/10,001 ≈
  9.999e-5. The observed statistic is always counted as a hit, so a
  zero p-value can never be emitted (regression-tested, Wave 3.0 P0).
- **MC uncertainty**: mc_se = sqrt(p(1−p)/(n_boot+1)) is stored in every
  artifact. At the correction thresholds used here (α ≥ 0.0018) the
  relative MC error is < 1% at p = 1e-4 (se ≈ 1e-5) and ≈ 10% at
  p = 0.01 (se ≈ 0.001); no reported decision is affected by MC noise.
- **Validity guards**: empty or non-finite deltas return
  `valid: false` with `signflip_p: null` and can never be counted as
  significant.

## 2. Multiple-testing families (declared, not post hoc)

| Family | Comparisons | Correction | Threshold | Result |
|---|---|---|---|---|
| Five-region primary (5 regions × 4 arms) | 20 | Bonferroni | 0.0025 | **12/20 survive** (6 beneficial / 6 harmful) |
| Architecture interaction (5 regions × 2 variants) | 10 | Bonferroni | 0.005 | **3/10 survive** (corrected, §3) |
| Physics six-arm primary (5 contrasts × 2 regions × 2 archs) | 20 | Bonferroni | 0.0025 | **3/20 survive**, all Charleston GRU, all increased error |
| Seven-family pilot (corrected inference) | 26 valid / 28 | Bonferroni | 0.0018 | 9/26 survive, all harmful, all REJECTED generators |
| Regional heterogeneity pairwise (mechanism-light) | 10 per arm | Holm | — | 8/10 pass; Cochran-Q p ≤ 1.4e-10 (all arms) |
| Severity dependence (mechanism-light) | 20 | Holm | — | effects attenuate with severity in harmful regions |

No cross-family pooling is performed anywhere; each family was declared
before its test. The pilot family (α = 0.0018) is a historical pilot and
is reported as such, never merged into the five-region family.

## 3. Architecture interaction — 4/10 → 3/10 (CORRECTED)

**Bug found and fixed** (commit b61d09c): `wave30_arch_interaction.py` v1
compared the *raw* sign-flip p against the Bonferroni threshold α = 0.005
instead of the adjusted p. Effect: SE-Australia/tail_aware (raw p =
0.0008, adjusted p = 0.008) was mislabelled a survivor.

**Verified v2 result** (`wave30_architecture_interaction.json`,
`audit_version: wave30_arch_interaction_v2`, flag-consistency check:
0 violations):

| Region | Variant | Interaction (mm) | p_raw | p_bonf | Survives |
|---|---|---|---|---|---|
| US Gulf | standard | −2.09 | 1e-4 | 0.001 | yes |
| US Gulf | tail_aware | −2.06 | 1e-4 | 0.001 | yes |
| SE Australia | standard | −1.29 | 4e-4 | 0.004 | yes |
| SE Australia | tail_aware | −1.22 | 8e-4 | **0.008** | **no** |
| Charleston | standard | +0.76 | 0.014 | 0.141 | no |
| Charleston | tail_aware | +0.82 | 0.010 | 0.096 | no |
| NY Bight | standard | −0.76 | 0.095 | 0.947 | no |
| NY Bight | tail_aware | −0.77 | 0.089 | 0.892 | no |
| Thames | standard | −2.41 | 0.179 | 1.000 | no |
| Thames | tail_aware | −2.35 | 0.185 | 1.000 | no |

Corrected statement: **3/10 survive; 8/10 interactions negative; every
survivor is in a region where augmentation helps.** The manuscript,
registry and status docs have been synchronised (the paper's previous
"9 of 10 negative" was also wrong — it is 8 of 10).

## 4. Exact vs Monte Carlo coverage

| Experiment | Unit counts | Method | Resolution floor |
|---|---|---|---|
| Five-region primary | 20–51 | MC (n > 16) | 9.999e-5 |
| Architecture interaction | 20–51 | MC (n > 16) | 9.999e-5 |
| Physics six-arm | 14–15 | **exact enumeration** | 1/16,384 ≈ 6.1e-5 |
| Seven-family pilot | 32–51 | MC (n > 16) | 9.999e-5 |

The physics experiment's exact enumeration is the strongest inference in
the project (no MC error at all); its three surviving Charleston GRU
contrasts all indicate **increased** error (harm), which is why the
manuscript must not describe the physical priors as beneficial.

## 5. Seed uncertainty

- **Generator seeds**: {42, 99} in every fidelity/augmentation cell;
  per-unit deltas are seed-averaged before testing. `seed_std_pct` is
  recorded per five-region comparison (range 0.01–1.70 pp). For 10 of 12
  five-region survivors |mean delta| exceeds 1.4 × seed_std; the two
  exceptions are the SE-Australia GRU cells (std: |delta| 1.36 vs
  seed_std 1.34; tail: 1.29 vs 1.57), whose effects are nonetheless
  consistent in direction across both seeds. Two seeds cannot support a
  variance claim, only a consistency check — stated as a limitation.
- **Downstream seeds**: {42, 99, 7} averaged in the five-region and
  physics experiments; the physics six-arm additionally averages 2 FM
  seeds × 3 downstream seeds per arm.

## 6. Event pairing and independence

- All primary tests are **within-region, within-architecture paired on
  identical POT event units** (event-ID equality is asserted in the
  interaction script; A0 and variant cells share the same window order).
- The five regional event-unit sets sum to 40 + 51 + 37 + 20 + 32 =
  **180 units**, but the manuscript must never call these 180
  independent storms: units within a region are declustered (extreme
  ≥ 168 h, moderate ≥ 120 h) rather than globally independent, units can
  share synoptic drivers across nearby regions, and clustering reduces
  the effective sample size. Consequently every p-value in the paper is
  a *within-region paired* statement; no test pools units across
  regions. Regional heterogeneity (Cochran-Q, Holm pairwise) is computed
  on region-level effect estimates, not on pooled units.
- The sign-flip null does not require unit independence, but clustered
  deltas would make p-values anticonservative; this is disclosed in
  Limitations.

## 7. Monte-Carlo p-value resolution audit

Smallest p-values in the primary families sit at the MC floor
(9.999e-5, e.g. Gulf and Thames survivors, p_bonf ≈ 1e-3). Since all
correction thresholds (0.0018–0.005) are ≥ 18× the floor, no decision
near the threshold can be an artifact of resolution. The SE-Australia
tail_aware interaction (p_bonf = 0.008 vs threshold 0.005) is a genuine
miss, not a resolution artifact — its raw p (8e-4) is 8× the floor.

## 8. Retrospective-test disclosure (unchanged, mandatory)

The 2022–2024 test years were used during protocol development in Waves
2.6–2.9. They are a **retrospective test split**, not an untouched
confirmatory holdout. Every downstream claim carries this disclosure;
the only potentially confirmatory data would be post-2024 years, which
are not available at current data coverage. P3 tail-calibration
candidates are frozen on train/validation before any test evaluation,
but the test years remain retrospective; P3 is classified exploratory.

## 9. Audit verdicts carried into the manuscript

1. Five-region result: 12/20 Bonferroni survivors, 6 beneficial / 6
   harmful — **verified**, retrospective disclosure attached.
2. Architecture interaction: **3/10** (not 4/10) — corrected and
   synchronised everywhere.
3. Physics six-arm: 3/20 survivors, all Charleston GRU harm — physical
   priors are **not** claimed beneficial; ERA5 meteorological
   conditioning, physics-inspired statistical regularisation and
   governing-equation PDE constraints are explicitly distinguished (the
   implemented method is regularisation, not PDE constraints).
4. Pilot 9/26 harm on REJECTED generators — pilot framing only.
5. 180 event units ≠ 180 independent storms — enforced wording.
6. Extreme stratum (n = 4–5 per region) — INCONCLUSIVE, never pooled
   into a tail claim.
