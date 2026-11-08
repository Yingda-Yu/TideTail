# Metric Definitions — TideTail Evaluation

**Status**: Reference document. All formulas implemented in `src/tidetail/evaluate.py`.

---

## 1. Statistical Units

Every metric is tied to a specific sampling unit. Mixing units without acknowledgement
is a common leakage source.

| Unit | Description | Used for |
|---|---|---|
| **Hourly cell** | One (forecast origin o, lead time h) pair | Overall MAE/RMSE, lead-time breakdown, station-wise |
| **Threshold event** | A hourly cell where y >= train-only threshold | Precision/recall/FPR/FAR detection |
| **Independent storm event** | Contiguous block of cells belonging to one meteorological storm | Peak amplitude / timing error, paired bootstrap |
| **Forecast window** | One complete (o → o+H) prediction | Useful for visualization; NOT an independent sample |

Critical rule for bootstrap CI: **sampling unit = independent storm events, not
overlapping forecast windows**. Overlapping windows are highly correlated —
bootstrapping on windows underestimates variance and overestimates significance.

---

## 2. Cell-Level Metrics

### MAE — Mean Absolute Error

```
MAE = (1/N) * Σ|y_i - ŷ_i|
```

where the sum runs over finite hourly cells only. Zero-finite-valid-cell return: `NaN`.

### RMSE — Root Mean Squared Error

```
RMSE = sqrt((1/N) * Σ(y_i - ŷ_i)^2)
```

Same finite-cell rule as MAE.

### Tail MAE / RMSE (train-only threshold τ)

```
Tail-MAE(τ) = mean(|y_i - ŷ_i|) over all finite cells where y_i >= τ
Tail-RMSE(τ) = sqrt(mean((y_i - ŷ_i)^2)) over all finite cells where y_i >= τ
```

τ is fixed from **train rows only**. Per protocol, both τ = q95_train and τ = q99_train
are reported. Cells where y_i < τ are excluded — we measure error on the events
that actually exceeded the training-tail threshold.

Rationale: the overall mean is dominated by benign cells (where tide dominates).
Tail metrics isolate the signal that augmentation is designed to improve.

---

## 3. Threshold Event Detection — Per-Hour-Cell

Let threshold τ be fixed from train rows only. Define:

| Cell category | Condition |
|---|---|
| True Positive (TP) | y >= τ AND ŷ >= τ |
| False Positive (FP) | y < τ AND ŷ >= τ |
| False Negative (FN) | y >= τ AND ŷ < τ |
| True Negative (TN) | y < τ AND ŷ < τ |

Only finite cells contribute to the contingency table.

### Precision

```
P = TP / (TP + FP)
```

Fraction of cells alarmed at or above τ that are actually at or above τ. NaN if
TP + FP = 0 (no alarms issued).

### Recall

```
R = TP / (TP + FN)
```

Fraction of cells actually at or above τ that are correctly alarmed. NaN if
TP + FN = 0 (no events exist in the held-out set).

### F1 Score

```
F1 = 2PR / (P + R)
```

Harmonic mean of precision and recall. NaN if both P and R are NaN or P + R = 0.

### False Positive Rate (FPR)

```
FPR = FP / (FP + TN)
```

**Fraction of non-event cells that were falsely alarmed.** Complement of specificity.
NaN if FP + TN = 0 (held-out set has no non-event cells — degenerate case).

**Important**: Do NOT confuse FPR with `FP / N` (false alarms per cell). The latter
is useful as an operational rate but is not FPR. Our code returns both — `fpr`
is the correct formula.

### False Alarm Ratio (FAR)

```
FAR = FP / (TP + FP)
```

Fraction of all alarms that are false. Complement of precision (FAR = 1 − P).
NaN if TP + FP = 0 (no alarms issued).

---

## 4. Event-Level Metrics

Computed per independent storm event. A held-out set contains M events; per-event
metrics have M values, which can then be aggregated or bootstrap-resampled.

### Peak Amplitude Error

```
PAE_event = |max(y_event) - max(ŷ_event)|
```

### Peak Timing Error

```
PTE_event = |argmax(y_event) - argmax(ŷ_event)|
```

in hours. First-maximum if ties exist. NaN if event has no finite y (or ŷ) values.

---

## 5. Structural Breakdowns

### Lead-time MAE/RMSE

Per horizon hour h ∈ {1, 2, …, H}:

```
MAE(h) = (1/N_h) * Σ|y_{o+h} - ŷ_{o+h}| across all forecast origins o
```

Identifies which lead times are hardest. Expected to increase with h (long-horizon
forecasts have higher cumulative uncertainty), but the augmentation hypothesis is
that tail-aware DDPM improves worst lead times disproportionally.

### Station-wise MAE/RMSE

Per gauge. Useful to see if improvements are universal or region-locked. Must be
used alongside held-out-station experiments if cross-gauge generalization is claimed.

---

## 6. Bootstrap Confidence Intervals

### Single-sample bootstrap on event-level values

For mean of a metric across M independent events (e.g., mean PAE):

```
For b = 1…B:
  Sample M values WITH replacement from {v_1, …, v_M}
  Compute mean of sample
Return 2.5% and 97.5% quantiles of the B bootstrap means
```

Requires **M >= 3 independent events** to produce non-degenerate CI. If M < 3,
returns NaN.

### Paired bootstrap for A vs B difference

For each storm k, compute both method-A and method-B values: (a_k, b_k). Difference
d_k = a_k - b_k. Then bootstrap resample the paired differences with replacement,
compute mean of each resample, take quantiles of the bootstrap distribution.

Advantages of paired bootstrap over two-sample: storms vary in severity; pairing
controls for this variation.

---

## 7. Zero-Event and Degenerate Cases

| Case | Return |
|---|---|
| Held-out set has zero independent storms | All event-level metrics: NaN. Bootstrap CI: NaN with n_events=0 |
| Held-out set has no cells >= τ | Tail MAE/RMSE: NaN. Recall denominator 0 → NaN. Precision/FAR/FPR/F1: NaN |
| All predictions are finite, no alarms issued | Precision: NaN (TP+FP=0). Recall/F1/FAR: NaN. FPR: 0 (FP+TN=N, FP=0) |
| Some cells NaN | NaN cells excluded from all denominators; n_finite counts reported |

Every metric function returns a structured dict when useful (threshold_metrics,
bootstrap_ci, lead_time_metrics), not a bare scalar. This prevents losing
denominator information when results are logged or tabulated.

---

## 8. Implementation Location

All formulas in `src/tidetail/evaluate.py`. Hand-computable unit tests in
`tests/test_evaluate.py` (added 2026-10-31, run as part of pytest gate).
