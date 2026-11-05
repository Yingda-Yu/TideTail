# Issue #6 CPU recovery: magnitude failure reproduced, G1 still blocked

2026-10-28. Owner task `3e5dffe0`, worktree `440c95e0`. CPU only, two threads,
no optimization, GPU queries, CUDA sampling, installs, downloads, paid APIs,
source-data changes or collaborator source-directory access.

## Scope and ancestry

Live issues #3/#4/#5/#6/#18 and all open PR metadata/discussions were read.
PR #20 head `e2d93fba65f670f9750aeae117ee9345c3ae1b47` is the parent of this
isolated repair. Draft targets `spartina/f1cff330`, not main, to isolate this
change. PR #19 is independently based on PR #17, with head
`3244c5c067f35c306d117cbcb2b45d48922ef9aa`, targeting
`feature/g2-real-data-baselines`. PR #17 head is
`584094cffd94bb7d806ca25d889c955dd483c0af`. The PR #20 tree lacks the baseline,
evaluator and split modules present in #19: ancestry alone does not make these
working trees composable. No changes from those active PRs were copied.

## A: independently reproduced negative result

The owner-produced checkpoint at the explicit path in `diagnose_ddpm_cpu.py`
is 435 KiB, SHA256
`dee516cf74b50a0ca8aff8ab05815c7ed36a32ea6c49b9dc81d5b532d39ebb68`.
Before load, verify its pinned size/hash, config training SHA
`5bc646676a08191a77e5d424f2cf52bb1af00c04`, all three historical source-file
hashes against Git objects, and freshly prepared data card equality including
migration/release/admission hashes, event IDs, station, split and train scaler.
Load with explicit `map_location="cpu", weights_only=True`, allowing only the
known PyTorch `TorchVersion` config class. No arbitrary pickle fallback.

| Result | Original CUDA artifact (preserved) | Independent CPU seed 99 |
|---|---:|---:|
| Minimum residual m | -16.909624 | -13.078981 |
| Maximum residual m | 7.940378 | 16.277262 |
| Sample population SD m | 4.473741 | 4.921983 |

CPU draws are exactly repeatable within CPU, but CPU and CUDA RNG streams differ:
maximum pointwise difference is 29.847111 m. This reproduces the magnitude defect,
not the exact original CUDA sequence. Eight generated sequences are not eight
independent storms. No augmentation utility or realism claim is made.

Training normalization is mean 0.1082523683 m, SD 0.2146118562 m, fitted to rows
`[0,70041)` before 69,946 overlapping 72h-history/24h-target windows. Inverse scaling
is `z * train_std + train_mean`, once, in metres of residual (total minus released
FES2022 tide), not absolute total water level. CPU generated SD is 22.95 times the
training **window-target** SD. Window weighting slightly changes that reference
relative to the full-row scaler. Independent storm count remains unknown;
1,449 threshold runs and 69,969 unique target hours are distinct denominators.
Full event IDs remain in the verified local data card; their list hash and source
hashes are in `diagnostics.json`.

Forward noising uses `sqrt(alpha_bar[t])*clean + sqrt(1-alpha_bar[t])*epsilon`.
Index 0 includes the first nonzero noise step; the previous alpha_bar there is 1.
Reverse epsilon mean equals the clean-prediction posterior mean; variance is
`beta[t]*(1-alpha_bar[t-1])/(1-alpha_bar[t])`, zero at index 0. Deterministic tests
verify endpoints, algebra, the full reference seeded sampler and noise variance.
Training actually targets unit Gaussian epsilon (not clean levels). Conditioning
uses only the preceding 72 observed residuals. No future weather/level covariates
enter this model; upstream tide/QC provenance remains a separate gate.

The cosine schedule's final beta is 0.999, alpha_bar approximately 5.93e-7.
Clean reconstruction amplifies epsilon error by
`sqrt((1-alpha_bar)/alpha_bar) = 1298.7754` at index 63. On the first eight fixed
train windows with seed 20261028, terminal epsilon MSE is 0.0057461, implying
21.1288 m clean reconstruction RMSE. The first reverse prediction spans
-61.146 to +49.678 m before the chain contracts to its still-bad final scale.
These are small train-only probes, not a general calibration study. See the
complete per-step trace and fixed-noise per-timestep diagnostics in `diagnostics.json`.
The evidence supports terminal sensitivity plus imperfect epsilon prediction;
it does not isolate training budget, architecture, BF16 or objective as a unique
root cause. No off-by-one, variance or inverse-scale equation defect was found.

### Targeted repair and limits

Five deterministic tests failed first (three missing nonfinite rejections, two
missing scale-screen tests); three mathematical tests already passed. Reject
nonfinite conditioning, epsilon and reverse states with the failing index.
Add a train-reference distribution screen for collapse, dispersion and mean shift.
Defaults accept SD ratios [0.1,5] and mean offset <=5 train SD: explicit engineering
policy, not a physical or learned criterion. The GPU entrypoint now records
`sample_scale.accepted` alongside preserved samples, after checkpoint/sample save.
Consumers must require this field before considering augmentation; passing it
still does not authorize augmentation. Historical summaries lack this field.

The original finite-only gate could treat absurd finite sequences as successful
sampling. That validation defect is repaired; **the learned generator remains
unrepaired and unapproved**. Equations and finite outputs are deliberately
unchanged. No clipping, changed schedule or post-hoc rescaling is presented as
scientific recovery. Before/after here means failed/passing validation tests,
not an invented numerical improvement. A schedule/objective change requires a
new reviewed training experiment after G1/G2.

## B: G1 follow-up, no frozen-data mutation

`g1_followup.json` rehashes inspected files against owner migration and release
inventory, checks timestamp alignment for mask probes, and audits event containment
in all six regions. Exact aggregate findings:

- HK event `pearl_river_hong_kong_event_01149`, inclusive [78903,78904], is assigned
  validation but crosses cut 78904. HK remains not formally admitted. No crossing
  threshold intervals were found in the other five regions; this is not storm
  independence. The existing #19 fix rejects overlapping **supplied groups**, not
  this released assignment.
- HK and Thames scaler counts 70,012 and 68,071 are exactly reproduced with the
  target-valid train mask. All eight columns' mean/std deltas are zero. Thus a
  common-target-mask denominator explains the old per-column-finite mismatch.
  Builder implementation and its fit-before/after-split contract remain unsigned;
  metadata booleans alone are insufficient proof. No metadata was rewritten.
- The parent migration manifest now exists and matches the inspected files.
  PR #19's historical missing-manifest observation is superseded for this check;
  content hashes do not establish licenses or vertical datum reconciliation.
- PR #19's midpoint-only `embargo_filter` is still unsafe for full support and is
  not in the PR #20 tree. For hourly issue origin o, the complete 72/24 support is
  `[o-72,o+24)`; a partition row block `[a,b)` permits only `o>=a+72` and
  `o<=b-24`. Same-station supports across partitions must not intersect; nearby
  storms may require additional scientific embargo. A midpoint gap is insufficient.
  Pilot train rows end 96h before the unused cut and all windows stay wholly in
  that block; no val/test windows exist here. This does not certify a three-way
  frozen split. Release 96/72 windows are a different experiment.
- ERA5 hindcast/observed future forcing lacks issue/valid/lead forecast-vintage
  evidence and cannot become operational conditioning. No forcing was added.
- Canonical storm grouping, cross-region independence, upstream QC/tide fit,
  license/redistribution, per-station datum and owner-reviewed split manifest
  remain blocked. PR #17 persistence/24h-seasonal results concern total water and
  overlapping forecast cells; this pilot generates residuals. The toy GRU and
  prediction-noise 'oversample' are not matched strong GRU/TCN and training A1.

B is an investigation/report only: adversarial G1 implementation changes are
left for the split-owning PR to avoid duplicating active work. Frozen source data
was not changed; no new G1 signoff is claimed.

## C: dependency-aware queue (proposed only, no downstream jobs submitted)

| Order / issue | Queue classification | Dependencies / human gate | Exact acceptance criteria |
|---|---|---|---|
| 1: G1 integrity, #3/#18 with #2 owner | CPU implementation READY; scientific signoff HUMAN-BLOCKED | Owner/data steward approves licenses, datums, canonical storms and exclusions | Fail-first tests for interval crossings incl. HK, full 72/24 support and extra storm embargo, mask/count contract and held-out perturbation invariance; deterministic event/time/station manifest hashes; owner signs exact frozen manifest; no raw-data rewrite without review. |
| 2: DDPM correctness, #6 | CPU diagnostics READY; training change BLOCKED | Review this diagnosis; event/data usage follows task 1 | Reproduce pinned checkpoint failure and equation/oracle tests; reject nonfinite/scale failures; choose schedule/objective remedy using train-only calibration; preserve raw before/after samples, units and hashes; no claim from clipping or finite loss. |
| 3: real-only GRU/TCN, #4/#5 | CPU design/tests READY; real experiments BLOCKED | Signed task 1; owner-approved compute budget | Freeze identical target, origins, 72/24 or 96/72 contract, missing mask, seeds, updates/examples and tuning; persistence/seasonal + real GRU/TCN + true training A1/A2; train-fixed q95/q99, event metrics and paired storm CI; artifacts and negative results per seed. |
| 4: bounded DDPM multiseed, #6 | GPU HOLD — DO NOT DISPATCH | Tasks 1–3 and sampling remedy; NEW explicit owner GPU/device/time/memory approval | Precommit seeds and matched budget, check live ownership/memory, explicit CUDA_VISIBLE_DEVICES, bounded steps/time, checkpoint/config/data/source hashes; sample-scale/spectra/tail/duration/copy diagnostics for every seed; retain failures. |
| 5: held-out augmentation, #5/#7 | HOLD | Tasks 1–4; owner approves frozen independent test protocol before access | Real-only validation/test, canonical independent storm count; A0/A1/A2/A3 matched forecaster/tuning budgets, fixed augmentation ratios; q95/q99 tail, peak/timing, detection denominators, lead/station breakdown, paired storm-bootstrap CIs; trace any paper claim via claims ledger, including nulls. |

Next owner decisions: review diagnosis/validation guard; assign G1 split/data
steward and signoff owners; choose forecast target/window contract; approve
train-only remedy criteria. Further GPU work remains unrequested and blocked.

## Reproduction and actual verification

Run from repository root, using existing dependencies. CPU sample load is trusted
and pinned; script refuses any checkpoint/hash/card mismatch. Repro needs the
unchanged local owner artifacts; they are not redistributed.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src timeout 60 python3 -m pytest tests/test_sampling_integrity.py -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src timeout 120 python3 -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src timeout 120 python3 scripts/diagnose_ddpm_cpu.py --output docs/experiments/issue6/cpu_recovery/diagnostics.json
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 PYTHONPATH=src timeout 60 python3 scripts/audit_recovery_g1.py --output docs/experiments/issue6/cpu_recovery/g1_followup.json
ruff check src tests scripts
python3 -m compileall -q src tests scripts
git diff --check
```

Actual gate: 19 passed, no deselection, no timeout, no optimization. Before-fix
5 failed/3 passed in `tests_before.txt`; after-fix full output `tests_after.txt`.
Full branch Ruff passes. First lint attempt had five unused noqa directives;
first G1 probe failed parsing blank missing values, then handled blanks as NaN.
Both failed logs are preserved. Neither failure altered source data. Full PR #19
suite was not rerun on this incompatible tree; its historical timeout/lint failures
are not certified resolved. Python 3.10.12, NumPy 2.2.6, PyTorch 2.13.0+cu130,
pytest 9.1.1. No dependencies changed. Current diagnostic script/data hashes and
final Git commit are recorded with the PR evidence; script is rerunnable.
