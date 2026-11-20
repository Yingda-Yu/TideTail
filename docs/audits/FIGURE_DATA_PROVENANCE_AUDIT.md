# FIGURE DATA PROVENANCE AUDIT — Wave 1.9

**Date**: 2026-11-02
**Lead auditor**: Wave 1.9 EXECUTE

---

## F6 CRITICAL ERROR — Forecast prediction ≠ Generator synthetic sample

**File**: `scripts/make_wave18_figures.py` lines 353-422 (now removed)
**Output**: `paper/figures/F6_residual_distributions.pdf`
**Conceptual error**: The figure claimed to show "Real vs Generated Residual Distributions." But:

- The "generated" curve was actually `pred_residual` from A3 .npz files
- `pred_residual` is the **FORECAST prediction** from the downstream GRU/TCN model
- It is NOT a synthetic sample from the Flow Matching generator
- The FM generator output is `model.sample(history)` — it was never called

**Correct interpretation**: The figure shows "How do forecast residuals from A3-trained GRU compare to observed residuals?" This is a **forecast distribution comparison**, NOT a **generator fidelity figure**.

**What it should be called**: "Forecast Residual Distribution Comparison (A0 vs A3-trained GRU)"

**What it should show for synthetic fidelity**: Call `FlowMatching.sample()` on the trained FM checkpoint and overlay synthetic residual sequences. Compare:
  - Real residuals (obs_residual)
  - FM synthetic residuals (generator.sample with conditioning history)
  - KS test between real and synthetic

**F6 verdict in Wave 1.8**: Misleadingly labeled. The analysis content was diagnostic (KS test between real and forecast pred residual distributions) but the wrong concept was labeled "synthetic fidelity."

## F3 ERROR — Calendar days mislabeled as independent storms

**File**: `scripts/make_wave18_figures.py` F3 function (now removed)
**Output**: `paper/figures/F3_storm_splits.pdf`
**Error**: Figure showed "Calendar-day pseudo-event counts per region" and caption implied these represented independent storm events. As Wave 1.9 demonstrates:
  - 4/5 regions have only **1 independent observ segment** in 2024 test data
  - Calendar-day counts (360, 360, 360, 360, 301) have no relationship to actual independent storm-event counts

**Correct F3**: Show independent-unit counts (168h-embargo) plus canonical TC overlays per region. Wave 1.9 counts: 1, 7, 3, 1, 10 independent units.

## Other Figure Audits

| Fig | Output | Wave 1.9 Audit | Verdict |
|---|---|---|---|
| F1 | Study Regions Map | Map is conceptually fine, but "A3 Δ%" annotations now exploratory | ⚠️ Update annotations |
| F4 | Tide/Residual Decomposition | Source data = prime-formal timelines | ✅ CORRECT |
| F8 | A0/A1/A2/A3 MAE bars | Numerical values correct, but "better/worse" labels imply significance | ⚠️ Re-label numerical improvement only |
| F9 | Tail 95/99 MAE | Same issue as F8 — numerical values correct | ⚠️ Re-label |

## Corrections Applied

1. F6 → Renamed to "Forecast Residual Distribution Diagnostic". Regenerate later from actual FM.sample() outputs.
2. F3 → Rewritten to show independent-unit counts from 168h-embargo analysis.
3. F1, F8, F9 → Annotations updated from statistical claims to numerical improvements with significance caveats.

## What Remains Correct

- F4 (decomposition): Data source verified, no claim of independence involved
- F1 (study regions): Map geometry correct, just need Δ% annotations updated
- All numerical data values: wave17_analysis_v2.json values are correct, just their interpretation was over-confident

---

**Audit complete.** Wave 1.8 figures contain NO numerical fabrications but several **misleading conceptual labels** that overstate statistical certainty. Corrections in Wave 1.9.
