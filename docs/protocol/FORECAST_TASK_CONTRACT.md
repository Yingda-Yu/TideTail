# Forecast Task Contract — v1 (Candidate)

**Status**: Candidate for scientific confirmation, not yet frozen for formal experiments.
**Owner**: All TideTail developers
**Last updated**: 2026-10-31

---

## 1. Main Task

**Target**: Observed total coastal water level above local chart datum, in metres.
**Forecast**: 96 hours of historical observations → 72 hours of future observations.
**Sampling**: Hourly UTC grid.
**Primary region scope**: ny_bight, thames_north_sea, southeast_australia, us_gulf_coast, south_atlantic_charleston (five admitted).
**Excluded regions**: pearl_river_hong_kong (license/datum/signoff pending).

This matches the frozen release contract in `database_manifest.json` (`window_contract: input_hours=96, horizon_hours=72`).

The historical pilot (Issue #6) used 72h→24h on de-tided residual only. That model is classified as **exploratory** and MUST NOT be compared directly against main-task metrics.

---

## 2. Target Definition

| Component | Column | Description | Source |
|---|---|---|---|
| Observed total water | `observed_total_water_level_m` | Raw tide + surge | Release `target_timeline_v1.csv` |
| Deterministic tide | FES2022 constituent forecast | Existing archive, deterministic | `tide_contract` in manifest |
| Residual | `water_level_m` = observed_total − FES2022_tide | De-tided surge residual metres | Pilot target (not main target) |

**Main task forecasts `observed_total_water_level_m`.**

DDPM generators may internally target residuals. If so, inverse transform before evaluation is:

```
final_generated_total = FES2022_tide[forecast_origin_o : forecast_origin_o+72]
                      + denormalize(generated_residual_72h)
```

where `denormalize(z) = z * train_std + train_mean` using train-only scalers
fit on the residual column of train rows only. The fixed deterministic tide
component ensures DDPM-generated residuals and all real-data baselines are
evaluated on the same final total-water target.

**Rationale**: Direct total-water forecasting requires the model to learn both
the strong tidal periodicity and the rare residual extremes, conflating two
distinct physical signals. Residual-only generators are easier to audit for
physical plausibility (amplitudes ~ ±1 m are common for surges, whereas total
water spans several metres with deterministic periodic components). But the
operational forecast the coastal manager needs is the total water level. The
protocol therefore permits residual generation with deterministic tide restore,
without claiming this is a "different" task.

---

## 3. Split Protocol

| Parameter | Value |
|---|---|
| Split method | Event-grouped chronological 8:1:1 |
| Embargo | **Full support interval**: train block `[a,b)` permits only forecast origins `o >= a+96` AND `o <= b-72` |
| Train-only scalers | Mean/SD fit on train rows only, applied without re-fit to all partitions |
| Train-only thresholds | q95/q99 from train rows only, applied to all partitions |
| Window generation | Stride-1 sliding windows. **No** imputation. Windows with any NaN target/tide/giving are excluded. |
| Missing mask | All models consume a boolean mask `present[i]` per row. Models may choose to mask-loss or ignore, but the mask is fixed and common across A0–A4. |

---

## 4. Evaluation Protocol

All A0–A4 compared on:

- Independent held-out **real storm events** only, not overlapping windows.
- Same downstream predictor architecture and optimizer budget.
- Same train/val/test split, same seeds, same augmentation fraction for A3/A4.

Metrics (see `docs/evaluation/METRIC_DEFINITIONS.md`):

| Metric | Unit | Scope |
|---|---|---|
| Overall MAE / RMSE | Metres | All hourly forecast cells on held-out events |
| Train-q95 tail MAE / RMSE | Metres | Held-out cells where `y >= q95_train` |
| Train-q99 tail MAE / RMSE | Metres | Held-out cells where `y >= q99_train` |
| Peak magnitude error | Metres | Per-event |
| Peak timing error | Hours | Per-event |
| Precision | Ratio | Threshold event detection |
| Recall | Ratio | Threshold event detection |
| False Positive Rate | FP/(FP+TN) | Negative example unit: non-event forecast cells |
| False Alarm Ratio | FP/(TP+FP) | Precision complement |
| Lead-time MAE | Metres | Per horizon hour (1..72) |
| Station-wise MAE / RMSE | Metres | Per gauge |
| Paired bootstrap 95% CI | Ratio | Independent storm events as sampling units |

---

## 5. Seed and Reproducibility

| Item | Value |
|---|---|
| Random seeds | `[0, 2024, 4242, 7777, 9001]` (five seeds; more available if budget permits) |
| Data split determinism | One frozen split manifest, hash-recorded, never re-generated per seed |
| Scaler determinism | One frozen train-only scaler per region |
| Augmentation determinism | Same random seed list applied to A0–A4, same base data |
| Git SHA | Every run records exact commit SHA |
| Data SHA | Every run records file hashes |
| Split SHA | Hash of split manifest, invariant across seeds |

---

## 6. What this Contract Does NOT Say

This document is a **development contract** — it unblocks code and test writing
but does not establish scientific validity. Before any publication claim on
these results:

- [ ] HK region permanently excluded in split manifest.
- [ ] License/redistribution for admitted regions confirmed.
- [ ] Vertical datum per station reconciled against local chart datum.
- [ ] Canonical meteorological storm event definitions signed off by data owner.
- [ ] ADR_TARGET_AND_HORIZON.md reviewed and signed.
- [ ] ERA5 forcing provenance (hindcast vs forecast vintage) finalized.

---

## 7. Relation to Historical Pilot (Issue #6)

| Aspect | Main task (this contract) | Issue #6 pilot |
|---|---|---|
| History | 96 h | 72 h |
| Horizon | 72 h | 24 h |
| Target | Observed total water (or residual + FES2022 restore) | Residual only, no FES2022 restore |
| Region | 5 admitted, multi-station possible | ny_bight, one station |
| Split | 8:1:1 chronological + embargo | No val/test, pure engineering training block |
| License | Signed G1 required | Engineering smoke only |

No metric comparison across rows is valid until a single contract is frozen.
