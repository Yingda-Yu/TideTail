# Wave 1 — Scientific Correctness Audit

**Audit date**: 2026-11-01
**Branch**: `trae/global-data-generative-research-20261031`
**Commit SHAs**: `244ec0e` (HEAD, pushed), `84b74ee` (prior experimental baseline)
**Auditor**: TRAE Solo Agent (autonomous)
**Scope**: W1-P0 audit + W1-P1 pre-fix bugs identified in scripts run 2026-10-31

---

## Executive summary

Five scientific bugs found in `scripts/run_a3_flowmatching_augment.py` and one in its augmentation utility contract. All affect the 120-record A3 exploratory run (`data/results/a3_flowmatching_augment_96_72.json`). The run is classified **COMPLETE_EXPLORATORY** — it may inform architecture choices but is **NOT eligible as final scientific evidence**.

The M2/M4 generative audit (`diffusion_ts_96_72.json`, `flow_matching_96_72.json`) and A0 baselines (`a0_baselines_96_72.json`) are **not affected** by these bugs — they use separate scripts with different evaluation paths.

---

## Bug W1-BUG-01 — A3 synthetic conditioning mismatch

**Severity**: CRITICAL
**Affected**: `scripts/run_a3_flowmatching_augment.py` lines 373–378
**Found at commit**: `84b74ee` (also present in `244ec0e`)

### Code
```python
synth_idx = np.random.default_rng(seed).choice(
    len(synthetic_tgt), size=n_synth, replace=False
)
rtr_t_a3 = np.concatenate([rtr_t, synthetic_tgt[synth_idx]], axis=0)
rtr_h_a3 = np.concatenate([rtr_h, rtr_h[:n_synth]], axis=0)  # ← BUG
```

### Problem
`synthetic_tgt[i]` was generated using `rtr_h[i]` as conditioning (per `generate_synthetic_residuals` at lines 148–176: FM is conditioned on ALL `rtr_h` rows, so row `i` of synthetic output was generated from row `i` of history). But the code pairs `synthetic_tgt[synth_idx]` with `rtr_h[:n_synth]` — the first n_synth history rows, **not** the matching rows.

This destroys the condition→target pairing for all A3 synthetic training examples. The forecaster learns `history_j → synthetic_target_k` where j and k are unrelated.

### Reproduction
```python
import numpy as np
np.random.seed(42)
N, H = 1000, 72
rtr_h = np.random.randn(N, 96).astype(np.float32)
synthetic_tgt = np.random.randn(N, H).astype(np.float32)
ratio = 0.25
n_synth = int(N * ratio)
synth_idx = np.random.default_rng(42).choice(len(synthetic_tgt), size=n_synth, replace=False)

# Buggy pairing
buggy_h = rtr_h[:n_synth]
buggy_t = synthetic_tgt[synth_idx]

# Test that paired rows don't correspond
match_count = sum(1 for i in range(n_synth) 
                  if np.array_equal(buggy_h[i], rtr_h[synth_idx[i]]))
assert match_count == 0  # Should be 100% for correct pairing, 0% for buggy
```

### Expected fix
```python
# Correct pairing
rtr_t_a3 = np.concatenate([rtr_t, synthetic_tgt[synth_idx]], axis=0)
rtr_h_a3 = np.concatenate([rtr_h, rtr_h[synth_idx]], axis=0)  # FIXED
```

### Regression test
Test: `tests/test_a3_conditioning.py::test_a3_synthetic_conditioning_alignment`
Protocol: For each synthetic index i, verify `rtr_h_a3[N+i]` equals the exact history row that conditioned `synthetic_tgt[synth_idx[i]]`.

### Consequences for old results
All 360 A3 records (5 regions × 3 ratios × 2 seeds × 2 models) use **misaligned conditioning–target pairs**. The exploratory MAE/tail results are **not scientifically interpretable**. They measure "how well GRU/TCN learns with misaligned pairs," not "does FM augmentation help."

---

## Bug W1-BUG-02 — Tail-MAE metric definition is wrong

**Severity**: HIGH
**Affected**: `scripts/run_a3_flowmatching_augment.py` lines 246–251 (`evaluate_forecaster`)
**Found at commit**: `84b74ee` (also present in `244ec0e`)

### Code
```python
# Tail errors: q95/q99 of absolute error
abs_err = np.abs(err)
tail_mae_q95 = float(np.mean(abs_err[abs_err >= np.percentile(abs_err, 95)]))
```

### Problem
This computes q95 of **prediction errors** and averages errors above that threshold. This is "worst 5% prediction-error MAE" — it has **nothing to do with water level extremes**.

The task contract (FORECAST_TASK_CONTRACT.md) requires:
> **Tail-MAE at extreme observed water levels**: MAE computed only at target time steps where observed total water exceeds q95 of the training observed total water distribution.

Current code labels it "tail_mae_q95" but delivers "worst_error_5pct_mae".

### Correct definition
```python
def tail_mae_at_observed_extremes(pred: np.ndarray, tgt: np.ndarray,
                                  train_q95_threshold: float,
                                  train_q99_threshold: float):
    """Tail-MAE at test time steps where observed total water ≥ train-only threshold."""
    abs_err = np.abs(pred - tgt)
    mask_q95 = tgt >= train_q95_threshold  # target IS observed total water
    mask_q99 = tgt >= train_q99_threshold
    tail_q95 = float(abs_err[mask_q95].mean()) if mask_q95.any() else np.nan
    tail_q99 = float(abs_err[mask_q99].mean()) if mask_q99.any() else np.nan
    return tail_q95, tail_q99
```

**Note**: This requires `train_q95` and `train_q99` thresholds passed from the data loader. Currently the A3 script only computes these for residual magnitude (`np.abs(rtr_t)`), not for observed total water. Since A3 operates on residual targets (not total water), a variant using residual threshold is needed, or A3 must be extended to track total water.

### Consequences for old results
All 120 records with `tail_mae_q95`/`tail_mae_q99` fields have **misleading labels**. The reported numbers are "worst prediction error" not "error at extreme water." Old tail results are **not comparable across augment groups** because the worst 5% of errors may fall on different water-level regimes for different models.

---

## Bug W1-BUG-03 — Peak amplitude error is global-max across all test windows

**Severity**: MEDIUM
**Affected**: `scripts/run_a3_flowmatching_augment.py` lines 241–244
**Found at commit**: `84b74ee`

### Code
```python
peak_amp_err = float(np.abs(np.abs(pred_np).max() - np.abs(test_tgt).max()))
```

### Problem
This takes the GLOBAL max of |prediction| across ALL test windows and ALL horizon steps, and compares with the global max of |target|. Single outlier prediction or target can dominate. Should be per-window then aggregated.

### Correct definition
```python
peak_pred_per_window = np.abs(pred_np).max(axis=1)  # per test window
peak_tgt_per_window = np.abs(test_tgt).max(axis=1)   # per test window
peak_amp_err = float(np.mean(np.abs(peak_pred_per_window - peak_tgt_per_window)))
```

Also, "peak" in coastal flooding should be **maximum positive observed total water**, not absolute residual. This is consistent with the storm-surge domain where positive surge (above predicted tide) causes flooding.

### Consequences
Peak metrics in A3 JSON are **not interpretable** — 120 records. A0 baselines use different peak evaluation (`run_a0_baselines.py`) and may use similar simplification — needs cross-check.

---

## Bug W1-BUG-04 — Unequal optimizer steps across augment groups

**Severity**: HIGH (violates AGENTS.md integrity rules)
**Affected**: `scripts/run_a3_flowmatching_augment.py` A1/A2/A3 sections
**Found at commit**: `84b74ee`

### Problem
All groups train for 30 epochs with same batch size (256). But training set sizes differ:

| Group | Train size (ny_bight example) | Optimizer steps per epoch | Total steps |
|---|---|---|---|
| A0 real-only | 69,970 | 274 | 8,220 |
| A1 oversampling | 210,030 | 821 | 24,630 |
| A2 physics augment | 139,940 | 547 | 16,410 |
| A3 0.1 ratio | 76,967 | 301 | 9,030 |
| A3 0.5 ratio | 104,955 | 410 | 12,300 |

A1 gets 3× more optimizer steps than A0. This is **training-budget leakage** — we cannot attribute MAE differences to augmentation quality when they may stem from more gradient updates.

### Required fix
Implement two explicitly named protocols per W1-010:

**Protocol F1 — Equal optimizer updates**: Same total optimizer steps, same architecture, same seeds, same batch size. Vary epochs to match total steps.

**Protocol F2 — Matched real-data exposure**: Same real examples/exposure, explicitly report synthetic fraction and total optimization cost.

Record optimizer steps, unique real samples, repeated real samples, synthetic samples per run.

---

## Bug W1-BUG-05 — A2 "physics augmentation" label is misleading

**Severity**: LOW (naming/documentation)
**Affected**: `scripts/run_a3_flowmatching_augment.py` A2 section label
**Found at commit**: `84b74ee`

### Problem
The A2 group is labeled "A2_physics_augment" and uses `residual_noise_augment`. The actual implementation adds **Gaussian noise to history input** — there is no physical simulation, no shallow-water equation, no wind/storm surge physics. The `augment.py` docstring honestly says "physically-motivated noise levels" (citing instrument uncertainty), but the experiment label says "physics augmentation" which overclaims.

### Resolution
Rename group to `A2_history_noise_augment`. Update augment.py docstring to be explicit: "NOT a physics simulation; models instrument uncertainty as Gaussian." Keep the implementation — it's a legitimate baseline, just not physics-guided.

---

## Bug W1-BUG-06 — A1 extreme-window criterion uses absolute residual

**Severity**: LOW (specification gap)
**Affected**: `scripts/run_a3_flowmatching_augment.py` A1 section line 329
**Found at commit**: `84b74ee`

### Code
```python
is_extreme = np.any(np.abs(rtr_t) >= train_q95, axis=1)
```

### Problem
A1 marks windows as extreme if ANY absolute residual value ≥ q95 of absolute residuals. This captures ALL extremes (both positive flooding surge and negative drawdown). For coastal flooding, we primarily care about **positive** extremes (water above predicted tide). The absolute-value criterion mixes both.

### Resolution
Use positive-only criterion: `is_extreme = np.any(rtr_t >= train_q95_pos, axis=1)` where `train_q95_pos = np.percentile(rtr_t[rtr_t > 0], 95)`. Report both positive-only and absolute variants explicitly.

---

## Bugs NOT found (good news)

- **M2 DiffusionTS 50-epoch audit** (`scripts/audit_diffusion_ts.py`, `scripts/train_generative_models.py`): Not affected. Independent evaluation through `quality.py`.
- **M4 FlowMatching 15-epoch runs**: Not affected. Same quality.py evaluator.
- **A0 baselines** (`scripts/run_a0_baselines.py`): Uses separate evaluation; peak metric may have similar global-max simplification but the MAE/tail there is computed correctly on real targets. Need cross-check.
- **NOAA/HURDAT2 downloads**: Verified by manifest, SHA256 recorded.
- **153 pytest tests**: Pass.
- **ruff clean**: Yes.

---

## Results classification

| Run | Records | Status | Eligible as evidence? |
|---|---|---|---|
| M2 DiffusionTS 50ep | 10 | **VALID** | YES — audit findings unaffected |
| M4 FlowMatching 15ep | 10 | **VALID** | YES |
| A0 baselines | 20 | **PENDING** | Cross-check peak metric |
| A3 augmentation | 120 | **COMPLETE_EXPLORATORY** | NO — bugs W1-BUG-01..06 |
| M3/M5/M6 smoke tests | — | **EXPLORATORY** | NO — not GPU-trained |
| P1 data downloads | 362 files | **VALID** | YES — manifest verified |

---

## Fix commit

All bugs must be fixed in one isolated branch/commit. Required deliverables:

1. Corrected `scripts/run_a3_flowmatching_augment.py`
2. New test: `tests/test_a3_conditioning.py` (fail-first for W1-BUG-01)
3. New test: `tests/test_a3_metrics.py` (fail-first for W1-BUG-02/03)
4. New test: `tests/test_a3_training_budget.py` (fail-first for W1-BUG-04)
5. Corrected evaluation schema
6. One-region bounded A3 diagnostic run (not 120 trainings yet)
7. Updated CLAIMS_LEDGER marking old A3 as exploratory
8. Updated A0 peak metric cross-check

---

## Related documents

- `AGENTS.md` — research protocol and integrity rules
- `docs/protocol/FORECAST_TASK_CONTRACT.md` — metric definitions
- `docs/status/WORKLOG.md` — task queue
