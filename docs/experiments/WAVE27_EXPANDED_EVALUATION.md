# Wave 2.7 — Expanded 2015–2024 Evaluation

**Status:** protocol implemented and boundary-verified; full two-region pilot
running. **No result is claimed until the run artifact exists.**

**Script:** `scripts/wave27_expanded_eval.py`
**Artifact:** `data/results/wave27_expanded_evaluation.json`

---

## 1. Why the protocol had to be rebuilt

Every existing checkpoint in the repository was trained on data that includes
2022–2024, so none of them can be evaluated under a 2022–2024 test split. The
expanded protocol therefore requires **retraining from scratch**: there is no
checkpoint reuse, and no result from Wave 1.7/Wave 2.0 transfers to it.

## 2. Split definition

| Split | Years | Role |
|---|---|---|
| Train | 2015–2020 | FM fitting and downstream fitting |
| Validation | 2021 | FM epoch selection only |
| Test | 2022–2024 | touched once, at the end |

**Window construction happens after the time split.** A window starting at `s`
is admitted only if both `[s, s+96)` (history) and `[s+96, s+168)` (target) lie
entirely inside one split. A 168 h embargo is enforced at both boundaries.

Verified boundary report (Charleston, from the smoke run, which uses the same
split builder as the full run):

| Field | Value |
|---|---|
| train last target end | 52,607 |
| val first history start | 52,608 |
| val last target end | 61,367 |
| test first history start | 61,368 |
| required gap | 168 h |
| `train_val_gap_ok` | true |
| `val_test_gap_ok` | true |
| train / val / test windows | 52,441 / 8,593 / 26,137 |

The off-by-one in the boundary check (`max + span < min`) was found and fixed to
`max + span − 1 < min`; a false failure here would have silently discarded valid
splits.

## 3. Leakage controls

* **Normalisation** is per-sample for downstream inputs and train-only for the
  physics/feature scalers. No validation or test statistic enters any transform.
* **Model selection** uses validation loss only (`fm_selected_epoch`); the test
  split never selects an epoch.
* **Thresholds** (POT q95) come from the frozen manifest computed on training
  years; test years are labelled, not re-thresholded.
* **Augmentation** draws synthetic windows only from the training split.
* **No future covariates.** The FM conditions on history only (see
  `WAVE27_PHYSICS_FM.md`). Future ERA5 is never read.

## 4. Event units

Independent units are POT q95 clusters with 168 h declustering from
`data/manifests/multiyear_declustering.json`, relabelled per year because
`cluster_id` restarts each January:

```python
label = -(year * 1000 + cluster_id + 1)
origin = ts[s + 96]
```

**Limitation (carried into the manuscript).** The canonical-TC overlay for
2022–2023 is not present in the existing prediction archives, so these units are
**POT-only**. TC-driven surges that do not exceed q95 in the residual series are
therefore missing from the unit count. This makes the event-level test
*under-inclusive*, not inflated — but it is a deviation from the Wave 2.0
POT+TC definition and must be stated as such.

## 5. Statistical treatment

* Paired bootstrap (30,000 resamples) and sign-flip permutation (30,000) over
  **event units**, not windows.
* Two tests in this pilot (two regions × one candidate), Bonferroni α = 0.025,
  reported alongside the raw p.
* Monte-Carlo resolution floor is 6.7 × 10⁻⁵ (bootstrap) and 3.3 × 10⁻⁵
  (permutation), both below α = 0.025, so the test is not resolution-limited.
* This pilot is **not powered** for any confirmatory claim at 2022–2024 event
  counts; it is a directional feasibility test.

## 6. Downstream comparison

Both arms train the same GRU architecture on a **fixed 8,000-window budget**
with 200 optimizer steps. The only difference is whether `n_aug = 2,000` Flow
Matching samples (ratio 0.25) are appended. Comparing at equal budget — rather
than at equal real-window count — is what makes the augmentation effect
interpretable.

## 7. Results

Run: `1442.3 s`, CPU, 15 epochs, 600 test windows, 2,000 augmentation samples,
8,000-window/200-step downstream budget.

| Region | Train / Val / Test windows | FM selected epoch | Fidelity | A0 extreme MAE | FM-augmented | Δ | paired 95% CI (m) | p (boot / perm) |
|---|---|---:|---|---:|---:|---:|---|---|
| Charleston | 52,441 / 8,593 / 26,137 | 13 (val 0.7740) | **REJECTED** (scale 3.97, W1 0.511, q95 4.01) | 0.1175 | 0.1279 | **+8.88%** | [+0.0035, +0.0170] | 0.0047 / 0.0045 |
| NY Bight | 52,441 / 8,593 / 26,137 | 13 (val 0.9669) | **REJECTED** (scale 4.76, W1 0.628, q95 8.53) | 0.1392 | 0.1613 | **+15.85%** | [+0.0183, +0.0262] | 1.0e-4 / 5.0e-5 |

Identical window counts for the two regions are expected, not a bug: both series
are gap-free over 2015–2024, so 52,608 h of training years yield
`52,608 − 167 = 52,441` windows, 2021 yields 8,593 and 2022–2024 yield 26,137.
Charleston has 40 independent test units; NY Bight has 51.

### 7.1 The pilot does not test H1

**Both Flow Matching models trained under the expanded protocol fail the
fidelity gate** (scale 3.97 and 4.76 against a gate of 0.5–2.0). The
augmentation is therefore an over-dispersed generator, and the downstream
degradation — +8.88% and +15.85%, both significant even at Bonferroni
α = 0.025 — is the expected consequence of training on bad synthetic data,
not evidence about whether *good* generative augmentation helps.

This is a **negative result about the pilot's readiness**, and it must not be
reported as "FM augmentation harms forecasting". The production Flow Matching
checkpoints reach scale 0.945–1.116; the expanded-protocol models were fitted
for 15 CPU epochs and do not, so the gap is a training-budget/regularisation
difference, not a property of the protocol or the family.

### 7.2 What the run does establish

1. The split builder is correct: 168 h embargo respected at both boundaries,
   no train/val/test window overlap, and independent unit counts (40, 51) that
   match the event census scale.
2. Validation-only selection works (best epoch 13 in both regions, with
   validation loss still falling — the model is under-trained, not over-fit).
3. Downstream budget matching is implemented and the paired event-level
   machinery returns finite CIs and p-values.
4. **A poorly fitted generator measurably harms the forecaster**, which is a
   useful calibration of effect size: the damage from over-dispersion is larger
   than any benefit observed in the corrected Wave 2.6 grid.

### 7.3 Next step (not run)

The expanded protocol needs the FM fitted to the production budget (or
early-stopped on validation *fidelity*, not validation loss) before it can test
H1. Fidelity-aware early stopping is the obvious control: select the epoch
minimising validation loss **subject to** scale ratio ∈ [0.5, 2.0]. That was
not implemented in this pilot.


## 8. Reproduce

```bash
python3 scripts/wave27_expanded_eval.py \
    --regions south_atlantic_charleston ny_bight \
    --epochs 15 --n-test 600 --n-aug 2000 \
    --downstream-n-train 8000 --downstream-steps 200
```
