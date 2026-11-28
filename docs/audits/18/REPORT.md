# Issue #18: independent CPU integrity audit

Date: 2026-10-24. Source task: `f8de0128`. Scope: existing repository and
authorized migrated Prime release only. No data downloads, GPU operations,
real-model training, paid APIs, sudo, original collaborator-directory access,
or changes to source data. Original paths inside JSON were read as provenance
strings only. No literature or publication-performance claims are made.

**Decision: NO-GO for a claim of leakage-free independent-storm performance or
for diffusion experiments dependent on signed-off G1.** Diagnostic rolling
forecasts can be reproduced, but do not establish that claim.

## Repository and review scope

Initial checkout was clean, branch `spartina/c00664ed`, commit
`554dfc2929ef221094ae52d251ac467611a42189`. It contains only
`assert_disjoint_groups`, five tests, protocol/roadmap and a CPU CI workflow;
no real experiment configuration or pipeline exists at that commit.

Open issues and PR metadata were read through the connected owner GitHub account
after `gh issue list --repo Yingda-Yu/TideTail --state open --limit 100` failed
because GH_TOKEN was unset. See [snapshot](evidence/github-snapshot.json).
Issue #18 records this audit; implementation work in #2–#7 and PR #13 was not
reimplemented. GitHub reads used `/issues?state=open&per_page=100`,
`/pulls?state=open&per_page=100`, and `/pulls/{14,15,16,17}/files?per_page=100`
under `https://api.github.com/repos/Yingda-Yu/TideTail`.

| Draft | Examined head SHA | Finding |
|---|---|---|
| #14 | `5b790e559a7b2e99c77d5cb91b892cbf8c325a32` | Data/QC/split toy scaffolding; original tests do not reject overlapping event intervals. |
| #15 | `584094cffd94bb7d806ca25d889c955dd483c0af` | Same head as #17, including real baseline script; not a separate independent implementation. |
| #16 | `b100649ed3219b688db330d4f4847f41c6e7603c` | Inventory/experiment template/ledger only; its ACL-blocked status is stale relative to the migrated copy. Template fields do not enforce runtime invariants. |
| #17 | `584094cffd94bb7d806ca25d889c955dd483c0af` | Real rolling baseline evidence exists, subject to limitations below. |

The isolated audit branch starts at #17's exact head. Its draft PR targets
`feature/g2-real-data-baselines`, making only the audit and split validation fix
reviewable without copying the active workstream into another main-target PR.
No existing branch was overwritten or merged.

## Findings and gates

| Gate | Result | Evidence and implication |
|---|---|---|
| Physical data existence | PASS | Migrated release exists at `/data/yingda/tidetail-data/prime-formal-2015-2024`. Files are hashed, not inferred from PR prose. |
| Migration provenance | INCOMPLETE | `MIGRATION_MANIFEST.json` claimed in #17 is absent at both the release root and owner's TideTail `data/` directory. A release `file_inventory_sha256.json` exists; matching release hashes establish content consistency, not the authorization/license chain for the migration. |
| License/datum sign-off | BLOCKED | Source-name, metre/Pa/m/s field names and source hashes exist. A reviewed source-license/redistribution and per-station vertical-datum reconciliation card was not established. Numeric metre fields alone do not prove datum compatibility. Original source paths were not opened. |
| Target definition | QUALIFIED | `water_level_m` is observed total minus deterministic FES2022 tide; PR17 evaluates `observed_total_water_level_m`. Do not call these residual metrics or pool raw absolute levels across unlike station datums. |
| Region admission | FAIL for all-six formal claim | Release explicitly excludes `pearl_river_hong_kong`, with `formal_training_authorized=false` and 134 missing calendar hours. Its computed metrics are audit diagnostics only; audit authorization does not change release admission. |
| Chronological preprocessing | PARTIAL | Recompute training quantiles/scaler summaries from `[0,70137)`; validation `[70137,78904)`, test `[78904,87672)`. Exact deviations are in evidence. Boolean `no_future_or_test_fit` is not proof of builder behavior; upstream builder source was not recovered. |
| Event atomicity | FAIL as general guarantee | Disjoint label strings alone do not establish independent storms or full-interval separation. Original helper sorts start indices without validating overlap, repeated IDs or interval validity. Four new adversarial cases failed before the isolated fix. |
| Window split containment | QUALIFIED | Release contract is 96-hour input / 72-hour horizon with strict containment. PR17 independently builds 72/24 windows wholly inside its test segment, without using released window indices or event-table assignments. This avoids direct train/test row overlap for those baselines, but is a different experiment. |
| Embargo | FAIL as general guarantee | `embargo_filter` compares event midpoint to protected rows. Probe accepts interval [0,100] against protected index 0 with embargo 1. It is not called by `partition_chronological`. The isolated fix rejects overlapping groups but does not add a lookback/horizon embargo policy. |
| Cross-region independence | UNVERIFIED | Region-prefixed threshold-run IDs are not canonical storm IDs; same-time or related storms need joint grouping. Separate per-region baseline calculations do not pool data, but no pooled/held-out-station generalization result is certified. |
| Operational covariates | BLOCKED | Release explicitly says historical hindcast only. ERA5 reanalysis/nearest-grid series lacks archived issue/valid/lead-time forecast evidence; future observed forcing cannot be operational conditioning. Gauge-only baselines do not consume it. |
| QC leakage | UNVERIFIED | QC helper estimates 3-sigma/IQR on the supplied record and uses retrospective linear interpolation. There is no API requirement to fit only training data or respect issue-time availability; splitting must precede calling it. Release says no imputation, so helper risk is not evidence that migrated files were imputed. |
| Extremes | INCOMPLETE | Event table describes train-q95 total-water exceedance runs, often brief tidal peaks. These are not independently verified storm clusters; no declustering sensitivity or canonical multi-region event mapping exists in examined code. q99 thresholds exist but PR17 does not report q99 utility. |
| Metric semantics | FAIL for event-level claim | MAE/RMSE/tail are forecast-cell weighted across overlapping windows. Peak errors are per-window argmax/max, not per independent storm. `false_alarm_rate=FP/N` is neither FP/(FP+TN) nor FP/(TP+FP); missing truth can count as a false positive. `bootstrap_ci` bootstraps scalar values, with no paired event-ID contract, and baseline script does not invoke it. |
| Baseline strength/fairness | INCOMPLETE | Persistence and period-24 seasonal are useful diagnostics. `oversampled_persistence` adds/averages prediction noise; it is not training-set oversampling A1. `SimpleGRU` is an explicitly toy finite-difference model, not a strong neural baseline. No runtime matched-budget enforcement is present. |
| Diffusion readiness | NO-GO | `ConditionalDenoiser` adds conditional biases to noisy input. In its training error `pred-X`, the clean X cancels; passing a shape/finite-loss smoke test does not establish learning the coastal distribution. No verified A3/A4 downstream result or gain exists. Memorization correlation checks skip constant chunks and do not prove non-memorization. |
| Baseline CLI status | FAIL | Successful `evaluate_baselines` result has no `status='ok'`, so CLI prints `Skip reason: None` even when saving metrics. No upstream result file was overwritten during this audit. |
| CI lint | FAIL at audited PR head | `ruff check src tests` reports 89 findings. Audit additions are checked separately. Do not describe the complete PR CI as green. |

Remaining defects are recorded for their active workstreams rather than expanding
this bounded fix into a replacement evaluator, QC system or model study.

## Reproduction commands and task log

No packages were installed. Python is `/usr/bin/python3` (3.10.12), NumPy 2.0.2,
pytest 9.1.1. `python` was unavailable; the initial `python -m pytest` attempt
failed with command-not-found. The corrected initial checkout command
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 -m pytest -q` passed 5 tests.
All subsequent Python commands set `CUDA_VISIBLE_DEVICES=''`,
`OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

Read-only snapshots were made with:

```bash
mkdir -p /tmp/tidetail-audit-c00664ed/{pr14,pr16,pr17}
git archive 5b790e559a7b2e99c77d5cb91b892cbf8c325a32 | tar -x -C /tmp/tidetail-audit-c00664ed/pr14
git archive b100649ed3219b688db330d4f4847f41c6e7603c | tar -x -C /tmp/tidetail-audit-c00664ed/pr16
git archive 584094cffd94bb7d806ca25d889c955dd483c0af | tar -x -C /tmp/tidetail-audit-c00664ed/pr17
git switch -c audit/18-cpu-integrity-c00664ed 584094cffd94bb7d806ca25d889c955dd483c0af
```

The first branch switch was blocked by the read-only external worktree metadata;
the same isolated branch operation succeeded with sandbox escalation. No
auto-review rejection occurred.

With the environment prefix above, exact commands are:

```bash
PYTHONPATH=/tmp/tidetail-audit-c00664ed/pr14/src python3 -m pytest /tmp/tidetail-audit-c00664ed/pr14/tests -q
PYTHONPATH=/tmp/tidetail-audit-c00664ed/pr17/src python3 -m pytest /tmp/tidetail-audit-c00664ed/pr17/tests -q
PYTHONPATH=src python3 -m pytest tests/test_audit_split_regression.py -q
PYTHONPATH=src python3 -m pytest tests/test_audit_split_regression.py tests/test_splits.py tests/test_leakage.py -q
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m pytest -q -k 'not denoiser_trains_and_generates'
python3 scripts/audit_prime.py --root /data/yingda/tidetail-data/prime-formal-2015-2024 --output docs/audits/18/evidence/prime-audit.json
PYTHONPATH=src python3 scripts/audit_probes.py > docs/audits/18/evidence/additional-probes.json
ruff check scripts/audit_prime.py scripts/audit_probes.py tests/test_audit_split_regression.py
ruff check src tests
git diff --check
```

Captured outputs are under `evidence/`. The first inventory attempt failed
when treating a scalar NPZ member as a vector; its traceback is preserved in
`inventory-first-attempt.txt`. The script now flattens preview arrays, preserving
their original shapes/dtypes. An initial inline probe JSON dump failed on a
NumPy bool; `audit_probes.py` makes the conversion explicit. Neither failure
altered the source data or licenses. These are audit-tool failures, not negative
model results.

The two full-suite invocations (frozen PR17 and final branch) were explicitly
timeboxed with SIGTERM, exit 143, while in the finite-difference denoiser test;
their partial dot output is preserved. Only the two processes verified by exact
command line and this worktree's cwd were stopped. The separately recorded
bounded suite excludes that one training test. A complete full-suite/CPU CI pass
is therefore not claimed, even if all bounded tests pass.

The audit script reads only the verified migrated root and refuses symlinks. It
hashes all local files, compares release-declared byte counts and SHA256s, audits
target/tide/forcing timestamps, missingness, split statistics and training-only
summaries, and reproduces rolling persistence/seasonal metrics independently of
the PR baseline/evaluator. It stores aggregate evidence and file hashes only;
no observation tables, checkpoint files or raw licensed data are committed.

## Verified inventory, metrics and test results

Machine-readable evidence: [prime-audit.json](evidence/prime-audit.json).
There are **350 files, 471,109,529 bytes**. All **294** entries in the release
inventory match both byte count and SHA256. The other 56 local files are hashed
but are not covered by that release inventory. This is a release consistency
check, not a license or migration-chain attestation.

All six target grids contain **87,672 hourly UTC rows**, from 2015-01-01 00:00
through 2024-12-31 23:00. No duplicate timestamps or non-hourly steps were found
on these derived target grids; this does not certify uninspected native sources.
Tide and forcing timestamps match the target grids in all six regions; forcing
numeric fields contain no nonfinite values. Missing flags equal nonfinite total
water-level counts: NY 0, Hong Kong 134, Charleston 0, Australia 31, Thames 2,296,
Gulf 0. Station IDs, source names, QC flag distributions, event-label counts,
manifest hashes and NPZ shapes/previews are included in the artifact.

Training q95/q99 total-water thresholds reproduce to floating-point precision
in all six regions. Residual-water scalers reproduce exactly. Other scalers
reproduce exactly in the four regions without missing training targets; in
Hong Kong and Thames, per-column-finite recomputation differs and stored counts
instead equal the finite residual-target training counts (70,012 and 68,071).
A common target-valid mask is a plausible explanation, **not verified here**;
the builder/mask contract must be inspected before calling these mismatches
leakage or certifying the complete preprocessing pipeline.

Event label sets are disjoint, but
`pearl_river_hong_kong_event_01149` crosses a chronological cut. No crossing
threshold-run intervals were found in the other five event tables. That does
not establish meteorological storm independence. Released NPZ windows use
96/72 and PR17 uses 72/24; their different valid-window counts are retained.

The following are **computed diagnostic rolling test-window metrics**, in
metres. They reproduce PR17's saved persistence/seasonal values; no generator or
trained neural model is evaluated. Seasonal means a fixed 24-hour repeating
history. No test-set tuning was done. Every window contains 72 observed history
hours and 24 finite target hours, stride one, fully within the test segment.
No normalization, imputation or meteorological conditioning is used in this
independent reproduction. High-tail MAE uses the stored training q95 cutoff.

| Region | Windows | Unique target hours | Persistence MAE / RMSE | Seasonal MAE / RMSE | q95 tail MAE persistence / seasonal |
|---|---:|---:|---:|---:|---:|
| ny_bight | 8,673 | 8,696 | 0.573643 / 0.709674 | 0.214874 / 0.267759 | 0.809387 / 0.202256 |
| pearl_river_hong_kong (excluded; diagnostic only) | 8,665 | 8,688 | 0.554570 / 0.699842 | 0.156113 / 0.195845 | 0.885180 / 0.155766 |
| south_atlantic_charleston | 8,673 | 8,696 | 0.649513 / 0.799473 | 0.223217 / 0.269391 | 0.886827 / 0.163382 |
| southeast_australia | 8,673 | 8,696 | 0.450931 / 0.560698 | 0.138380 / 0.163936 | 0.766760 / 0.121819 |
| thames_north_sea | 7,688 | 7,918 | 1.749002 / 2.147482 | 0.597974 / 0.707293 | 2.637719 / 0.432711 |
| us_gulf_coast | 8,673 | 8,696 | 0.193148 / 0.255496 | 0.111768 / 0.163535 | 0.212780 / 0.138038 |

The overall denominator is windows × 24 forecast cells, so target hours can be
counted repeatedly. q95 tail-cell counts in table order are 15,275, 14,588,
10,595, 12,630, 10,065 and 15,167. **Independent storm counts remain unknown**;
no storm-level uncertainty, significance, A1/A3/A4 benefit or operational skill
is established by these values. Stronger baselines and declustered events remain
required. The poorer persistence numbers and all qualifiers are retained.

| Verification | Actual result | Output |
|---|---|---|
| Initial main checkout | 5 passed | Captured terminal result; source SHA above |
| Frozen PR14 | 21 passed | [pr14-pytest.txt](evidence/pr14-pytest.txt) |
| New regression before fix | 4 failed, 1 passed | [regression-before.txt](evidence/regression-before.txt) |
| Regression + split/leakage tests after fix | 13 passed | [regression-after.txt](evidence/regression-after.txt) |
| Final bounded CPU suite | **54 passed, 1 deselected**, 387.28 s | [bounded-pytest.txt](evidence/bounded-pytest.txt) |
| Frozen PR17/full final suite | Stopped at timebox, exit 143 | `pr17-pytest.txt`, `final-pytest.txt`; no complete pass claimed |
| Audit additions lint | Passed | [audit-lint.txt](evidence/audit-lint.txt) |
| Inherited `src tests` lint | 89 findings, failed | [pr17-lint.txt](evidence/pr17-lint.txt) |
| `git diff --check` | Passed | Final local verification |

The fix validates event intervals and duplicate IDs before partitioning and
rejects overlap, while preserving adjacent disjoint intervals. It does not
repair the migrated event table, sign off event independence, or implement a
forecast-window embargo. Full CPU CI still needs the slow deselected test and
inherited lint findings resolved by the owning workstream.

## Prioritized next experiments

1. **Freeze G1 first:** review source licenses, migration provenance, units/datum,
   region admission, canonical storm IDs across regions, complete-event purging,
   shared chronological cuts and an input/horizon-aware embargo. Hash the final
   row/event/station split and all exclusions. Verify scalers/thresholds by
   held-out-data perturbation tests; do not fit preprocessing across splits.
2. **Freeze real-only baselines:** retain last-value persistence, fixed seasonal
   lags, and training-only climatology. Compare on validation first with a strong
   GRU/TCN under fixed budgets/seeds. Use the same target, input/horizon,
   missing-data mask and forecast origins for all comparisons. Keep 72/24 smoke
   separate from the released 96/72 experiment. No extra test-based selection.
3. **Freeze extreme evaluation:** report train-q95/q99 cell errors, lead/station
   breakdowns, independent event peaks/timing, event detection and denominators;
   report zero-event folds as undefined. Cluster related threshold runs/stations
   into storms, then compute paired storm-level uncertainty, not window-level CIs.
4. **Conditional diffusion pilot only after G1/G2 sign-off:** A0 real-only, A1
   actual training duplication/balanced sampling, A2 justified perturbation,
   A3 actual standard conditional diffusion, then A4 tail-aware objective.
   Use train-only histories and available-at-issuance covariates, or explicitly
   segregated hindcast/oracle forcing. Match capacity, steps, input examples,
   tuning and compute; freeze augmentation fractions and multiple seeds.
5. **Ablations and nulls:** vary augmentation ratio, q95/q99 tail weighting,
   conditioning and optional residual/tide reconstruction/spatial constraints.
   Audit spectra, duration, tail mass, multi-gauge lags and nearest-neighbor
   copying before downstream utility. Preserve null/negative outcomes. Advance
   H1/H2 only through `paper/CLAIMS_LEDGER.md` with reviewed artifacts and
   independent event counts; this audit does not establish either hypothesis.
