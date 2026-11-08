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

### 5.1 Corrected run

Being rerun with `CUDA_VISIBLE_DEVICES=""` on CPU. Results will be inserted here
from the refreshed `data/results/wave27_physics_fm.json`; no value is typed by
hand.

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
