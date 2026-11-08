# Wave 2.7 — Physics-Guided Flow Matching

**Status:** ERA5 conditioning verified for all five regions; five-variant pilot
run in progress on two verified regions. **No physics result is claimed until the
run artifact exists.**

**Script:** `scripts/wave27_physics_fm.py`
**Artifact:** `data/results/wave27_physics_fm.json`
**Readiness artifact:** `data/results/era5_readiness_matrix.json`
**Causality tests:** `tests/test_physics_causality.py` (4 tests, CPU)

---

## 1. ERA5 readiness matrix

`scripts/verify_era5_readiness.py` audits every downloaded partition against the
forecast timeline (`verified_at = 2026-11-05T13:57:01Z`). Grid is
`0.25 degree regular_ll, 17x17 points`.

| Region | Files | Hours | Months failed | Continuity | Anchor → nearest cell | Integrity |
|---|---:|---:|---:|---:|---:|---|
| ny_bight | 120 / 120 | 87,672 / 87,672 | 0 | 119 / 119 | 4.50 km | OK |
| us_gulf_coast | 120 / 120 | 87,672 / 87,672 | 0 | 119 / 119 | 9.10 km | OK |
| south_atlantic_charleston | 120 / 120 | 87,672 / 87,672 | 0 | 119 / 119 | 11.70 km | OK |
| southeast_australia | 120 / 120 | 87,672 / 87,672 | 0 | 119 / 119 | 11.00 km | OK |
| thames_north_sea | 120 / 120 | 87,672 / 87,672 | 0 | 119 / 119 | 6.60 km | OK |

`all_regions_integrity_ok = true`. Variables are `u10` and `v10` in m s⁻¹, and
`msl` in Pa (`expver` 0001); wind speed is derived as √(u² + v²).

**Representativeness caveat.** A 0.25° cell spans roughly 28 km at these
latitudes. Charleston's anchor is 11.7 km from its assigned cell centre, so
near-coastal MSL and wind are cell averages that do not resolve the harbour
scale. This is an unresolved limit of the forcing, not a defect to be papered
over; it bounds how much a physics prior can be expected to contribute.

---

## 2. What "causal" means here

Every physics covariate is summarised **only over the 96 h conditioning history**
`s .. s+96`, where `s` is the forecast origin. The 72 h target horizon `s+96 ..
s+168` is never read. `tests/test_physics_causality.py` asserts this directly:

1. Mutating a window's target horizon does not change its own features.
2. Mutating a window's history does change them.
3. The `s = 0` window does not produce empty slices (the earlier `[s-96, s)`
   interval bug is regression-tested).
4. `standard` and `tail_aware` modes expose zero physics dimensions.

This makes the experiment **reanalysis-conditioned but causal**: at forecast
time one would need a forecast of MSL and wind, not the reanalysis. It is
therefore *not* an operational forecast result, and the manuscript must say so.
A retrospective future-forcing variant (which would leak) was deliberately not
implemented.

---

## 3. Variants

All five use identical splits, identical FM architecture (`VelocityTCN`), and
identical downstream forecaster budget.

| Variant | Physics input dim | Prior |
|---|---:|---|
| `standard` | 0 | history only |
| `tail_aware` | 0 | history only, tail-weighted FM loss |
| `pressure` | 4 | `msl_last`, `msl_mean24`, `msl_tend24`, `msl_mean72` |
| `wind` | 5 | `u_last`, `v_last`, `speed_last`, `speed_mean24`, `speed_mean72` |
| `combined` | 9 | both |

Feature definitions (all causal, `hi = s + 96`, `last = hi - 1`):

* `msl_last` — sea-level pressure at the forecast origin, Pa
* `msl_mean24`, `msl_mean72` — mean pressure over the last 24 h / 72 h, Pa
* `msl_tend24` — 24 h pressure tendency `msl[last] − msl[hi−25]`, Pa
* `u_last`, `v_last` — 10 m wind components at the origin, m s⁻¹
* `speed_last`, `speed_mean24`, `speed_mean72` — wind speed √(u²+v²), m s⁻¹

**Physical assumptions (explicit).** A storm surge prior asserts that low
pressure and onshore wind over the preceding 24–72 h raise coastal water level
through the inverse-barometer effect and wind-driven setup. The chosen features
are the minimal set that can express this at 0.25° resolution: pressure level,
pressure tendency (deepening), and sustained wind. They carry **no** direction
projection onto the coastline normal, so onshore/offshore wind is not
distinguished; this is a known omission. Bathymetry conditioning is not included
because GEBCO was not downloaded — that variant is unimplemented, not negative.

Normalisation uses **train-split-only** statistics, computed once from
`build_features(train_starts, ...)` and reused unchanged for validation and test.

---

## 4. Protocol

1. Train each variant's FM on the training split (2000–2022 timeline), fit on the
   full training window set unless `--n-train` is given.
2. Sample `n_aug` synthetic (history, target) pairs, keeping the pairing intact
   (each synthetic target is generated from its own history window).
3. Train downstream GRU forecasters on a **fixed** 8,000-window budget, identical
   across A0 and every variant, so the comparison isolates augmentation.
4. Fidelity gate (`tidetail.quality.evaluate`) on 300 held-out windows × 3 draws.
5. Downstream extreme-window MAE delta versus A0.

Earlier pilot (8,000-window FM-fitting cap, superseded) produced scale ratios
4–5 and REJECTED fidelity for every variant. That was initially read as a
training-budget artefact. It was not: Wave 2.7 later traced it to a sampling-path
defect (E26) that affected every Wave 2.7 generative pilot. The current run fits
the FM on the full training set **with the sampling path corrected**.

---

## 5. Results

### 5.0 Superseded full-data run (sampling-path defect, E26)

The 6,636.3 s full-data run wrote A0 Charleston 0.1013 and NY Bight 0.1155 with
downstream deltas −1.60/−1.16/−0.58/+2.31/+9.44% and
+3.35/+3.39/+3.05/+4.69/+4.60%, but every one of its ten variants failed fidelity
at scale 3.19–4.09. A controlled diagnostic proved that the failure came from
the shared sampler `sample_windows` feeding **raw metre-scale history** to a
generator fitted on per-sample-normalised history and never inverting the
normalisation: scale 4.95 raw versus 1.17 normalised on the identical model.

**Those numbers are withdrawn in full** and are preserved only as
`data/results/wave27_physics_fm_prefix_sampling_bug.json`. The fix is in
`sample_windows`, guarded by
`tests/test_physics_causality.py::test_sample_windows_normalises_history_and_de_normalises_output`.

### 5.1 Corrected run (COMPLETE, 2026-11-08)

Rerun with `CUDA_VISIBLE_DEVICES=""` on CPU under systemd unit `tidetail-wave27`
(run `wave27_20261108T081124Z_16955577`, exit 0, 7,448 s, code SHA `1106ecbb`,
command: `python3 -u scripts/wave27_physics_fm.py` with defaults — epochs 10,
n-test 200, n-aug 2,000, generator draws seeded 42/43/44, downstream GRU seed
42, artifact sha256 `53f575208c33939bc4989f9e6e17c9005fe1e2aa1ca934f493ba91544610c93f`).
All values below are read from the refreshed
`data/results/wave27_physics_fm.json`; no value is typed by hand.

**Fidelity: every variant is INCONCLUSIVE in both regions** (scale 0.869–1.077,
W1 0.024–0.139). The E26 fix restores generator fidelity for all ten
variant × region cells.

Downstream extreme-unit MAE delta vs A0 (paired per-unit sign-flip test on the
14/15 extreme units; A0 GRU extreme MAE: Charleston 0.1074, NY Bight 0.1173):

| Variant | Charleston Δ | p | NY Bight Δ | p |
|---|---:|---:|---:|---:|
| standard | −2.20% | 0.052 | −2.02% | 0.037 |
| tail_aware | −2.09% | 0.056 | −2.04% | 0.031 |
| pressure | −1.95% | 0.045 | −1.88% | 0.030 |
| wind | **+1.25%** | 0.024 | −1.85% | 0.026 |
| combined | +0.09% | 0.813 | −2.00% | 0.029 |

**Interpretation (P0-4 rules applied):**

1. Ten variant × region comparisons at a *single* generator draw-set and a
   *single* downstream seed (42); nominal p < 0.05 in 8/10 cells but **nothing
   survives Bonferroni (α = 0.005)**. These deltas are exploratory.
2. The sign pattern is inconsistent for physics arms: the wind-prior arm
   *harms* Charleston (+1.25%, the only nominally significant harm) while the
   same arm improves NY Bight (−1.85%); the combined arm is a no-effect in
   Charleston (+0.09%, p = 0.81). **No physics arm beats `standard`
   consistently** — physical penalties/auto-calibrated λ provide no robust
   downstream gain over plain conditioning.
3. These arms are **physics-inspired statistical regularization** (train-only
   OLS penalty on implied surge; auto-calibrated frozen λ), not
   governing-equation PDE constraints; era5 conditioning is retrospective
   reanalysis, not an operational forecast product. No bathymetry is used
   (GEBCO not downloaded).
4. Consistency check across protocols: on the same-day corrected
   expanded-year protocol (Wave 2.8 `wave27_expanded_evaluation.json`), which
   uses from-scratch training on a leakage-controlled multi-year split,
   standard-FM augmentation *harmed* extreme-event MAE (+0.86%/+0.92%).
   The −2% improvements here come from the older primary-split protocol with
   a pre-trained augmentation budget and a single downstream seed; they do not
   replicate under the stricter split. The expanded-year result supersedes
   this table for augmentation-value claims; this table is retained as the
   corrected-E26 physics-variant comparison under its own protocol.

---

## 6. Reproduce

```bash
python3 -m pytest tests/test_physics_causality.py -q
python3 scripts/verify_era5_readiness.py
python3 scripts/wave27_physics_fm.py \
    --regions south_atlantic_charleston ny_bight \
    --epochs 15 --n-test 300 --n-aug 2000 \
    --downstream-n-train 8000 --downstream-steps 200
```
