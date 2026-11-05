# ADR: Forecast Target and Horizon

**Status**: Candidate, awaiting scientific signoff
**Date**: 2026-10-31
**Decision makers**: TideTail research principals
**Supersedes**: (none)

---

## Context

The TideTail dataset provides two natural forecast targets and two competing
window contracts:

### Target choices

1. **Observed total water level** (`observed_total_water_level_m`). Includes
   deterministic FES2022 tide plus storm surge. This is the operational variable
   coastal managers need for evacuation decisions. But it has a strong
   deterministic periodic component, and models might appear to perform well by
   simply learning the tide.

2. **De-tided surge residual** (`water_level_m = observed_total - FES2022_tide`).
   Carries the operationally most urgent signal (storm surges), is zero-centred
   and approximately stationary on short timescales, but is not directly useful
   to coastal managers without the tide re-added.

### Horizon choices

1. **96h history → 72h forecast** (release contract). Full 168-hour window
   length matches standard NOAA/USGS forecasts. Operational relevance.
2. **72h history → 24h forecast** (Issue #6 pilot). Shorter for engineering
   feasibility. Already explored; results not publishable against main contract.

---

## Decision

**Target**: Use `observed_total_water_level_m` as the **primary evaluation target**.
Generative models (DDPM) may internally target residuals, but MUST restore the
deterministic FES2022 tide with the correct time alignment before evaluation.

**Horizon**: 96h → 72h (release contract), frozen for all A0–A4 comparisons.

### Why not residual-only as primary target?

- Residual-only forecasts cannot be operationally interpreted without tide
  restoration. An evaluation on residuals produces numbers that cannot be
  compared to coastal engineering criteria (which are specified in total water
  level above chart datum).
- Residual-only evaluation does not preclude good generative models — DDPM
  targeting residuals is easier to audit (amplitudes ~±1 m are realistic,
  whereas total water spans several metres with tidal periodicity). But the
  evaluation step must restore tide to reach the operational target.

### Why not both as co-primary?

- Running two full evaluation pipelines doubles the work without proportionally
  more scientific value. The residual result is operationally opaque; the total
  water result is the one that matters. Residual-only numbers can be reported
  as an ablation, not as a co-primary outcome.

---

## Consequences

### Positive

- Results are directly interpretable to coastal managers and flood risk analysts.
- Comparable to NOAA/USGS forecast evaluation standards (which use total water).
- Residual-targeting generators still possible — just add one deterministic step.

### Negative

- Total water includes deterministic tide, which dominates forecast error at
  benign lead times. Metrics like overall MAE will be inflated by tide-learnable
  errors. However, tail q95/q99 metrics isolate the surge signal where it matters.
- Model capacity split: some capacity learns tide, some learns surge. This is
  unavoidable — even residual models must consume tide-conditioned history.

### Mitigation

Report both overall metrics (reflects full operational performance) and tail
metrics (focuses on the signal that augmentation is designed to improve). The
tail metrics are the primary judge of H1.

---

## What needs human signoff

1. Per-station vertical datum reconciliation — ensure `observed_total_water_level_m`
   is referenced to the correct local chart datum, not a global ellipsoid or
   geoid. This affects all absolute-level claims.
2. FES2022 tide model applicability per region — the release manifest specifies
   this but no independent QC of tide fit has been performed.
3. Era5 forcing provenance — if used at all, must verify forecast vintage
   (hindcast is not usable as operational conditioning).
4. HK region status — confirm whether pearl_river_hong_kong is permanently
   excluded or conditionally admissible following datum/license signoff.

---

## Evidence Basis

- Release manifest `database_manifest.json`: `window_contract: {input: 96, horizon: 72}`
- `data/train_only_metadata_v1.json` per region: scaler fits and q95/q99
- Issue #6 pilot: residual-only, 72h→24h, 27,169-param DDPM, magnitude failure
  (SD ratio ~22.95×). Not publishable as anything other than a diagnosis.
