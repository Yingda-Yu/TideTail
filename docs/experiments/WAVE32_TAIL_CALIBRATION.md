# Wave 3.2 P3 — Bounded Tail-Calibration Study (exploratory extension)

**Status: COMPLETE (2026-11-28, 1,833 s CPU, PID 1874613). Classification:
exploratory extension — the 2022–2024 test years are retrospective; NOT a
confirmatory test.** Artifact: `data/results/wave32_tail_calibration.json`.
Script: `scripts/wave32_tail_calibration.py` (commit `aaaf7b3`).

## 1. Bounded design (frozen before test evaluation)

No new model families. Six arms, contrasting regions only
(Southeast Australia — augmentation helps; Thames/North Sea — augmentation
harms), downstream GRU only (declared a priori), frozen calendar protocol
(2015–2020 / 2021 / 2022–2024), event-level paired sign-flip tests vs. A0.

| # | Arm | Type |
|---|---|---|
| 1 | standard FM | existing generator |
| 2 | tail-aware FM | existing generator |
| 3 | quantile-calibrated FM | post-hoc per-position ECDF mapping onto train targets (NEW) |
| 4 | fidelity-selected FM | 6 draws/window, keep first draw inside a train-derived plausibility band (window std within train [q05,q95]; mean within 3 MAD) (NEW) |
| 5 | real-event oversampling | duplicate train windows with peak ≥ train q95 (real targets) |
| 6 | history-noise control | real targets, history + N(0, 0.1·per-window std) (σ pre-registered, not tuned) |

**Stage A (validation-only, frozen):** candidate fidelity evaluated on 2021
validation windows; a candidate is adopted iff
|log(scale)|+|log(q95r)| improves ≥10 % over raw FM. Decisions written to
disk before any test evaluation. Outcome: **fidelity-selected FM adopted in
SE-Aus seed 99 only; quantile-calibrated FM adopted in Thames seed 99
only** — i.e. validation-fidelity improvements were marginal and
seed-inconsistent. Both arms were nevertheless evaluated for both seeds
(pre-registered), and every per-seed result is reported.

**Multiplicity:** 2 regions × 6 arms = 12 GRU paired comparisons;
exploratory; Bonferroni α = 0.05/12 = 0.0042 reported alongside raw p.

## 2. Results (extreme-event MAE delta vs. A0 GRU; mean over generator seeds; per-seed range)

### Southeast Australia (augmentation helps here)

| Arm | Δ% (seed 42 / seed 99) | p (42 / 99) |
|---|---|---|
| standard FM | **−1.04** (−0.47 / −1.62) | 0.035 / 0.0001 |
| tail-aware FM | −0.99 (−0.71 / −1.27) | 0.003 / 0.0002 |
| quantile-calibrated FM | −0.44 (−0.07 / −0.81) | 0.26 / 0.0001 |
| **fidelity-selected FM** | **−1.22** (−0.82 / −1.62) | 0.0012 / 0.0002 |
| real-event oversampling | **+7.62** (+7.14 / +8.10) | 0.0001 / 0.0001 |
| history-noise control | +0.60 (+0.35 / +0.84) | 0.0002 / 0.0001 |

### Thames / North Sea (augmentation harms here)

| Arm | Δ% (42 / 99) | p (42 / 99) |
|---|---|---|
| standard FM | +3.24 (+3.38 / +3.11) | 0.0001 / 0.0001 |
| tail-aware FM | +3.25 (+3.38 / +3.11) | 0.0001 / 0.0001 |
| quantile-calibrated FM | +3.13 (+3.29 / +2.96) | 0.0001 / 0.0001 |
| fidelity-selected FM | +3.05 (+3.26 / +2.84) | 0.0001 / 0.0001 |
| real-event oversampling | +0.43 (+0.48 / +0.38) | 0.33 / 0.43 (ns) |
| history-noise control | −0.07 (+0.04 / −0.18) | 0.70 / 0.31 (ns) |

## 3. Findings (negative results preserved, no further tuning)

1. **Validation-fidelity calibration does not rescue augmentation in the
   harmful region.** Both calibration candidates improve validation
   composite modestly (adopted once each, on different seeds), yet the
   generative arms remain seed-robustly harmful in Thames (+2.8…+3.4 %,
   all p = 0.0001, well beyond the exploratory Bonferroni α = 0.0042).
   The harm is not a tail-calibration artifact that post-hoc mapping can
   remove.
2. **In the beneficial region, the fidelity-selected candidate is the best
   generative arm (−1.22 % vs. −1.04 % standard)** — consistent with its
   validation adoption — but the margin is within seed variability, so it
   is a weak ordering, not a proven improvement.
3. **Real-event oversampling is harmful where augmentation helps**
   (+7.6 %, SE-Aus, both seeds p = 0.0001) and null where it harms
   (+0.4 % ns, Thames) — the opposite interaction pattern from FM
   augmentation, and a caution against assuming "more real extremes" is
   always better.
4. **The history-noise control is essentially null in Thames (−0.07 % ns)
   and marginally harmful in SE-Aus (+0.60 %, both seeds significant)** —
   crude non-generative augmentation does not reproduce the FM benefit in
   the beneficial region.
5. **Fidelity-versus-utility gap confirmed prospectively:** validation
   fidelity ranking did not predict downstream ranking in Thames
   (calibrated variants scored best on validation composite yet remained
   most harmful), replicating the main paper's central gap under a
   frozen-selection design.

## 4. Scope and stop

- The study stops here by mandate: no generator re-tuning beyond this
  bounded set; the negative Thames result is preserved as-is.
- Because the test years are retrospective, this study is classified as an
  exploratory extension and does not alter the paper's primary claims; the
  manuscript reports it in §6.11 (Figure F12) with this classification.
