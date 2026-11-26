# Wave 3.0 Statistical Correctness Audit — 2026-11-26

Scope: re-audit of historical statistical conclusions per Wave 3.0 P0-04.
All numbers below are recomputed from artifacts on disk, not copied from
narrative documents.

## 1. Zero-p-value Monte Carlo defect (P0-03) — FIXED

`paired_event_stats()` in `scripts/wave28_physics_fm.py` (also re-exported
and used by `wave28_expanded_eval.py`, `wave28_gen_benchmark.py`,
`wave28_physics_analysis.py`) previously computed

```python
p = float(np.mean(stat >= abs(obs) - 1e-12))
```

Two failure modes:

1. **NaN input → spurious p = 0.** When `delta` contains NaN, `obs = NaN`,
   every comparison `stat >= NaN` is False, and the mean is exactly 0.0.
   This manufactured a "highly significant" p-value out of a generation
   failure.
2. **Zero p from finite input.** With finite n_boot Monte Carlo, a p-value
   of exactly 0 is not admissible; the minimum reportable two-sided
   sign-flip p is 1/(n_boot+1).

Fix applied (commit at head of this wave):

- NaN / Inf / empty input → explicit `{"valid": False, "signflip_p": None}`
  invalid state; callers must not report significance.
- Monte Carlo branch → `p = (b + 1) / (n_boot + 1)` including the observed
  statistic as one guaranteed hit.
- Exact enumeration branch (n ≤ 16) keeps `b / 2^n`; the identity and its
  full negation always match, so the floor is 2/2^n, never zero.
- Exchangeability assumption of the sign-flip null is documented in the
  docstring.
- Regression tests added in `tests/test_wave28_physics.py` covering:
  all-zero diffs, all-same-direction diffs, exact small-n floor,
  MC minimum p floor, NaN/Inf input, empty input, event-ID misalignment,
  and key-based (architecture/seed) pairing.

Test status at fix time: 318 passed (full suite, CPU).

## 2. Re-audit of the "11/28 significant harm" benchmark claim (P0-04)

Source: `data/results/wave28_gen_benchmark.json` (Wave 2.8, 2 regions,
7 families, GRU+TCN, 28 downstream comparisons).

Recomputed classification (`data/results/wave30_corrected_inference.json`):

| Item | Old claim | Corrected |
|---|---|---|
| Comparisons counted | 28 | 26 valid + 2 invalid |
| Survivors of Bonferroni α=0.0018 | 11/28 | **9/26** |
| Direction of survivors | all harmful | all harmful (unchanged) |
| Survivor generators | mixed REJECTED | ddpm_cond, diffusion_ts (Charleston only), timegrad |

The two removed cells are `ny_bight/diffusion_ts/{gru,tcn}`:
`extreme_unit_mae = NaN`, `mean_delta = NaN`, old `signflip_p = 0.0`.
These are **generation failures** (Diffusion-TS diverged; Wave 2.7 already
established the x0-spectrum NaN mechanism), not statistical evidence of
harm. Counting them inflated the harm count from 9 to 11.

Corrected statement for the paper: *all surviving Bonferroni-corrected
downstream effects in the seven-family pilot are harmful and originate
exclusively from generators that failed the fidelity gate; no
fidelity-eligible generator (FM, tail-FM, WGAN-GP) produces any
significant downstream effect in either direction.*

## 3. Per-unit data availability

The Wave 2.8 benchmark did **not** persist `per_unit_mae` arrays (only
aggregate deltas + fitted stats), so the affected comparisons cannot be
re-inferred from stored means alone. Disposition:

- The 2 invalid cells are declared generation failures — no rerun needed
  to justify exclusion; Diffusion-TS NY Bight divergence is documented in
  `wave27_model_recovery.json` and the benchmark warnings.
- The 26 valid cells: corrected p-values are derived from the stored MC
  hit counts (`b = round(p_old × 10000)`), which is a lossless re-map of
  the same statistic; direction and ordering are unchanged (corrected p is
  monotone in old p).
- The Wave 3.0 stratified re-evaluation (P2) will regenerate all cells and
  now persists per-unit arrays + event IDs (fix applied to
  `wave28_gen_benchmark.py` and `wave28_expanded_eval.py`), so this audit
  gap cannot recur.

## 4. Other historical claims re-checked

| Claim | Status | Evidence |
|---|---|---|
| "707 events" | Withdrawn in Wave 2.7; union is 679 regional units | `wave26_event_set_audit.json` |
| Charleston "5/48 significant" | Withdrawn in Wave 2.7 (bootstrap p inflation + MC floor) | `docs/status/LATEST_PROGRESS.md` Wave 2.7 section |
| E26 scale ratios 3.97/4.76 | Withdrawn; per-sample normalisation fix committed `9b790a6` | regression tests `test_sample_windows_*` |
| Six-arm physics: Charleston 3 regularization contrasts significant (harmful, mm-scale) | Verified intact: exact enumeration, no zero p, no NaN in `wave28_physics_analysis.json` (scan clean) | this audit §1 scan |
| Physics primary contrasts (20 comps, α=0.0025) | No false zeros possible (exact branch, n=14 ≤ 16) | `wave28_physics_analysis.json` |

## 5. Actionable code changes shipped with this audit

1. `paired_event_stats` fix + 8 regression tests (§1).
2. `wave28_gen_benchmark.py` and `wave28_expanded_eval.py` now persist
   `per_unit_mae` + event IDs in every downstream cell.
3. `wave28_expanded_eval.py` gained `--resume` with cell-level completion
   checks, protocol-matched A0 reuse, and atomic JSON writes.
4. `sample_windows` device handling made robust (`net=None` stub path);
   leftover debug prints from the interrupted Wave 2.9 session removed.

## 6. Residual limitations

- The 26 valid pilot p-values remain single-season (first 200 test
  windows) and retrospective; they will be superseded by the Wave 3.0
  stratified re-evaluation, not silently reused.
- Sign-flip null assumes per-unit delta symmetry about zero; clustered
  units (same storm) reduce effective sample size and are disclosed as
  such in the analysis docs.
