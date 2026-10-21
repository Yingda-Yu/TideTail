# Issue #12: CPU data-integrity utilities

These utilities implement the bounded synthetic-data scope of issue #12. They
do not complete the real-source audit, define real storm groups, freeze a real
data split, or authorize training under the full G1 roadmap.

`tidetail.data.ObservationRecord` requires station ID, timezone-aware UTC
`datetime`, and water level in meters. Optional `event_id` identifies the same
event across every gauge. `validate_records` rejects missing/nonfinite/out-of-bound
levels, invalid IDs, and duplicate or decreasing timestamps within a station.
Input can be grouped by station. It is never silently sorted before validation.
Empty input is valid for validation/gap/imputation utilities, but cannot be split.

`chronological_split` returns a `ChronologicalSplit` with `train`, `val`, `test`,
and `excluded` tuples. Cuts are based on counts of distinct timestamps across all
gauges, using floor(n * train_frac) and floor(n * (train_frac + val_frac)).
All observations at the same timestamp share a temporal partition. Fractions
must be positive and sum to one within 1e-12. After each cut, the interval
`[cut, cut + embargo_days)` is excluded (default seven days). If an event crosses
a cut or touches an embargo, all of its rows across all stations are excluded.
This conservative policy can change retained fractions substantially. Empty
retained partitions raise `ValueError`; reduce the embargo only with scientific
justification or use a longer record. The existing `assert_disjoint_groups`
checks retained groups. With unknown event IDs (`None`), only temporal separation
can be checked; callers must supply canonical storm IDs for event independence.
`excluded` accounts for every dropped row, including whole-event purges.

```python
from tidetail.data import chronological_split, make_synthetic_records
from tidetail.metrics import tail_rmse

records = make_synthetic_records(days=60)
split = chronological_split(records, .6, .2, .2)
train_levels = [r.water_level for r in split.train]
# Toy fixture check only; no empirical forecasting claim.
score = tail_rmse([1., 3.], [1.5, 2.5], train_values=train_levels)
```

Split raw observations **before** interpolation, normalization, threshold fitting,
or creating windows. Choose an embargo sufficient for subsequent history and
target horizons. This API handles raw observation separation; it does not audit
downstream windows or future covariates. No scalers or window generation are
implemented here.

`gauge_gaps` returns station/start/end/elapsed-hours records for adjacent timestamp
gaps strictly larger than the threshold (default 24 hours). It also accepts rows
with missing values; it measures timestamp gaps, not runs of missing levels.
`impute_linear` fills existing internal None/NaN rows by elapsed seconds per
station and leaves observations, ordering, event IDs, and timestamps intact. It
does not insert rows for absent timestamps. It rejects extrapolation and anchors
with differing event IDs. Use it separately within each partition, after reviewing
gaps. It is two-sided retrospective interpolation and must not supply operational
forecast inputs where its right-hand anchor is unavailable at issuance.

Metrics accept matching nonempty finite 1D arrays; missing observations must be
handled explicitly before evaluation. MAE/RMSE weight observations equally.
`tail_rmse(y_true, y_pred, q=.95, *, train_values=None, threshold=None)` requires
exactly one training reference: finite training values for a NumPy linear
quantile, or a frozen train-derived scalar threshold. This deliberately prevents
the two-argument call from fitting a cutoff on held-out targets. Tail membership
is `y_true >= threshold`, including ties; its denominator is selected time points.
An empty tail returns NaN. Callers are responsible for training provenance of the
reference data/threshold.

`event_peak_error(y_true, y_pred, threshold)` computes mean absolute peak magnitude
error, weighting each contiguous true exceedance run equally. Predicted and true
peaks are compared within the same true run; predicted peaks outside the run are
not included. Threshold equality counts as an exceedance. No observed events
returns NaN. Pass one chronologically ordered station and continuous segment at
a time; this metric does not compute peak timing, false alarms, canonical storm
counts, or confidence intervals (those belong to issue #5).

`make_synthetic_records` uses a local NumPy RNG with seed 0 and generates fictional
meter observations on `[2020-01-01 UTC, start + days)`. Defaults: three stations,
30 days, 60-minute cadence. Event IDs are shared daily fixture groups, **not**
real independent storms. No source dataset, license, or downloaded data is used.
Fixture provenance is the function, parameters, seed, dependency versions, and
commit SHA; fixture records are generated in memory and never committed.

CPU verification from the checkout (no package installation needed):

```sh
PYTHONPATH=src /usr/bin/python3 -m pytest tests/ -q
ruff check src tests
```

Tests cover validation, temporal boundaries, equal-time multi-gauge alignment,
whole-event purges, embargo accounting, missing data, irregular-time interpolation,
deterministic fixtures, exact metric arithmetic, and zero-event evaluation.
No GPU code, new dependencies, dataset acquisition, or empirical experiments are
part of this change. Research specifications and paper claims are unchanged.
