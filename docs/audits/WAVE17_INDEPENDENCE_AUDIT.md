# Wave 1.7 independent-event audit — Issue #5

Audited source: `86169e1cd219db11e0c9befe7eb0b0bf659e6f72`, draft PR #23.
This audit supersedes the scientific acceptance statements in WAVE1_7_ACCEPTANCE.md.

190 archived prediction files verified and SHA256 recorded in
WAVE17_INDEPENDENCE_AUDIT.json. All 120 A3 comparisons are retained as descriptive
window-weighted MAE differences, without selecting a winning configuration.
No training was rerun. No independent storm confidence interval is justified yet.

| Region | Test windows | Adjacent overlapping target pairs | Test threshold excursions |
|---|---:|---:|---:|
| NY Bight | 8601 | 8600 | 231 |
| Charleston | 8601 | 8600 | 181 |
| Southeast Australia | 8601 | 8600 | 196 |
| Thames | 6968 | 6958 | 312 |
| Gulf | 8601 | 8600 | 75 |

Excursions are train-q95 total-water runs, not canonical storm identities. They
must not be counted as independent storms. The source manifest describes a
chronological 8/1/1 split; a separate storm database does not prove that these
training and prediction windows consumed storm-group splits. Storm boundary
and cross-station grouping remain unverified. Independent storm n is UNKNOWN,
not zero. The saved predictions contain no event IDs, window IDs, station IDs or
timestamps. For one archive per region, observations match reconstruction from
the current source window indices exactly; source hashes are recorded, but this
does not establish a historical frozen storm split.

`scripts/wave17_analyze.py::paired_bootstrap` samples window indices. Its narrow
intervals and PR's “72/72 significant” are not independent-event evidence. The
archive actually has 120 A3 comparisons across both architectures. Cluster
bootstrap must first map verified issue/valid times to canonical storm IDs,
merge correlated multi-station storm observations, and report equal-event paired
effects and event counts. Nonstorm windows require a separately declared estimand.
Do not silently substitute threshold excursions or arbitrary blocks for storms.

Code uses 8000 optimizer updates, batch cap 256, Adam 0.001 for trained arms;
archive metadata reports 8000 steps. Short last batches and different augmented
pool sizes mean equal updates need not imply equal examples. Seeds 42/99 are
reported, but model construction precedes `torch.manual_seed(seed)` inside
`train_forecaster`: initialization is not controlled by the nominal arm seed.
FM training and tuning compute are additional. “Best A3” is test-selected and
optimistic; report the full grid or choose using validation before a new test.

Double-detiding repair is present in both loaders: `water_level_m` is used
as residual. Current source residual + tide matches observed total water with
maximum error <= 8.9e-16 m. Further, for identical added tide,
`(pred_residual + tide) - (obs_residual + tide) = pred_residual - obs_residual`.
Therefore tidal dominance cannot by itself explain differences in unmasked
MAE/RMSE. Tide can affect correlation, total-water tails and peak timing, which
need separate diagnostics. The reported 20–30%/40–60% variance contributions,
ENSO explanation, and causal dilution explanation have not been measured here.

Environment: nvidia-smi cannot communicate with driver; no GPU launched. Process
inspection is namespace-limited and cannot exclude host TRAE/Codex jobs. No
existing worker located in this checkout; no scheduler launched. CDS config is
absent. Existing ERA5/GEBCO manifests were located, but licenses, hashes and
issue-time availability have not been validated; mere presence does not clear
W18-T3. Never treat future reanalysis as issue-time weather.

Reproduction:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/wave17_independence_audit.py \
  --predictions /data/yingda/TideTail/data/predictions/v2 \
  --source /data/yingda/tidetail-data/prime-formal-2015-2024/derived/prime_coast \
  --output docs/audits/WAVE17_INDEPENDENCE_AUDIT.json
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 -m pytest -q
```

The script reads trusted local NumPy archives with object metadata; do not use
it on untrusted pickle-bearing archives. JSON contains hashes, station IDs,
package versions and derived metrics, no raw observations or checkpoints.

Validation: baseline local CPU suite passed 268 tests (1 warning, 50.47s).
Added adversarial fixture checks that overlapping threshold excursions never
become canonical storms or a storm CI, and an empty archive creates no claims.
Targeted audit tests: 2 passed. Ruff passed for both new Python files; diff check
passed. Local data availability explains why local success cannot substitute for
the failing remote CPU gate, which must be addressed separately.

Full CPU recheck: 269 passed, 1 warning in 55.95s (collection preceded addition
of the adversarial fixture); final targeted file: 2 passed in 0.38s. No known
local test failures. Warning is pre-existing tensor-to-scalar conversion.
