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

### 7.0 Superseded first run (sampling-path defect, E26)

The first run (`1442.3 s`) produced scale ratios 3.97 and 4.76 and downstream
deltas +8.88% / +15.85%. Those numbers are **withdrawn**. A controlled
diagnostic on an identical trained model showed the cause: the shared sampler
`wave27_physics_fm.sample_windows` was handing **raw metre-scale history** to a
generator fitted on per-sample-normalised history, and returning its output
without inverting the normalisation. Scoring that path gives scale 4.95 against
1.17 for the normalised path (the production benchmark sits at 1.05). The defect
is fixed in `sample_windows`, locked by
`tests/test_physics_causality.py::test_sample_windows_normalises_history_and_de_normalises_output`,
and the pre-fix artifact is retained as
`data/results/wave27_expanded_evaluation_prefix_sampling_bug.json`.

The split, embargo, leakage and window-count findings in §2–§4 are structural
and are **not** affected by the defect.

### 7.1 Corrected run (E26 fix applied)

Completed 2026-11-08 (`2341.6 s`, CPU, `CUDA_VISIBLE_DEVICES=""`), artifact
refreshed in place at `data/results/wave27_expanded_evaluation.json`
(pre-fix copy retained as `..._prefix_sampling_bug.json`).

| Region | scale ratio | W1 | autocorr RMSE | q95 exceed ratio | status | downstream Δ vs A0 | p_perm | p_boot | units |
|---|---|---|---|---|---|---|---|---|---|
| Charleston | 1.147 | 0.019 | 0.043 | 1.60 | INCONCLUSIVE | **+0.86%** | 0.016 | 0.011 | 40 |
| NY Bight | 1.097 | 0.033 | 0.038 | 1.70 | INCONCLUSIVE | **+0.92%** | 0.0023 | 0.0013 | 51 |

**Reading.** The E26 repair restores generator fidelity to the regime of the
production primary-split benchmark (scale ≈ 1.1, W1 ≈ 0.02–0.03): the previous
3.97 / 4.76 verdicts were harness artefacts, as diagnosed. What the corrected
run establishes is an **honest negative downstream result**: under the
leakage-controlled 2015–2020/2021/2022–2024 protocol, standard-FM augmentation
does not improve extreme-event MAE — it degrades it slightly, by +0.86%
(Charleston) and +0.92% (NY Bight), and the paired event-level tests attribute
even these small harms to signal, not noise (p_perm 0.016 / 0.0023; n = 40 / 51
independent units). The generator remains only INCONCLUSIVE on fidelity (q95
exceedance ratio ≈ 1.6–1.7, i.e. synthetic tails are too heavy), so this is
"augmentation from a partially-fidelious generator", not "augmentation from a
fully calibrated generator" — the Wave 2.8 P2 six-cell comparison
(`scripts/wave28_expanded_eval.py`, GRU+TCN, standard+tail-aware, multi-seed)
is the follow-up that varies both dimensions.

### 7.2 What the protocol establishes regardless of the fidelity verdict

1. The split builder is correct: 168 h embargo respected at both boundaries,
   no train/val/test window overlap, 52,441 / 8,593 / 26,137 windows, and
   independent unit counts (40 Charleston, 51 NY Bight) that match the event
   census scale.
2. Validation-only selection is implemented and runs.
3. Train-only preprocessing and augmentation are enforced.
4. The downstream budget and event-level paired machinery return finite CIs.

### 7.3 Next step (not run)

If the corrected generator still fails the fidelity gate, fidelity-aware early
stopping is the obvious control: select the epoch minimising validation loss
**subject to** scale ratio ∈ [0.5, 2.0]. That is not implemented in this pilot.


## 8. Reproduce

```bash
python3 scripts/wave27_expanded_eval.py \
    --regions south_atlantic_charleston ny_bight \
    --epochs 15 --n-test 600 --n-aug 2000 \
    --downstream-n-train 8000 --downstream-steps 200
```
