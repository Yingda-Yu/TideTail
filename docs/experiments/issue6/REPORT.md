# Issue #6 GPU conditional diffusion pilot — prepared, training blocked

Task f1cff330, 2026-10-25. Isolated branch `spartina/f1cff330`, based on
`554dfc2929ef221094ae52d251ac467611a42189`. Scope: [issue #6](https://github.com/Yingda-Yu/TideTail/issues/6).
The task's explicit GPU-only override supersedes the issue's original CPU-pilot
wording. The cancelled a519bc52 task was inspected read-only, never resumed.

## Actual outcome

**No GPU selected, no gradient training on any device, zero optimizer steps,
no trained checkpoint, loss CSV, trained samples or training CPU measurement.**
Preparation and CPU forward/sampler tests succeeded. Training is blocked because
unreserved status of the apparently idle cards cannot be established from the
available records. Owner confirmation was requested; none received at report
creation. Do not equate repeated zero utilization with permission to use a card.

Initial sandbox `nvidia-smi` failed to communicate with the driver and sandbox
process visibility was restricted. Authorized host checks succeeded. Physical
RTX3090 GPUs 3 and 4 were repeatedly idle with no compute processes: GPU3 24 MiB,
0%, 28 C; GPU4 18 MiB, 0%, 26–27 C. Both have 24,576 MiB total VRAM. GPU0 was
excluded. GPUs 1/2 have owner workloads; 5–9 have collaborator workloads.
Host process inspection found no other TideTail trainer; Worker, Gateway,
llama-server and all other users' processes were left running. No scheduler
`squeue` executable or authoritative reservation record was identified from
processes, owner locations, `/tmp` lock names or `/var/lock`. These checks do
**not** certify a GPU as unreserved. See `hardware.txt` for a final timestamped
snapshot and ownership summary. Earlier observations are retained in the tool
transcript; no historical GPU accounting database was available.

## Review and data admission

Read README, protocol, roadmap, open issues, PR #14–19 metadata and files, PR15's
actual toy denoiser, PR17 implementation and PR19 audit report. PR15 uses per-step
bias/finite differences and does not implement the required trainable denoiser.
This branch builds from main without copying or modifying those draft branches.
PR19 head reviewed: `3244c5c067f35c306d117cbcb2b45d48922ef9aa`.

Only `/data/yingda/tidetail-data/prime-formal-2015-2024` was opened for source data.
Original paths embedded in metadata were treated as strings. No collaborator's
original dataset was accessed. A migration manifest actually exists **one level
above** the release at `/data/yingda/tidetail-data/MIGRATION_MANIFEST.json`;
this qualifies PR19's earlier missing-manifest finding. Its destination matches
the authorized owner root. Selected source files match both release and migration
size/SHA256 entries; admission metadata matches migration, and the region
manifest matches admission. Symlinks are refused.

Only admitted **ny_bight**, station `NOAA_8518750_The_Battery`, is used. HK and all
other regions are excluded from this single-station pilot. Target is release
`water_level_m` residual (observed total minus deterministic FES2022 tide), in
metres. Source identifiers are NOAA CO-OPS; reviewed source-license/redistribution
and station-datum sign-off are still incomplete. Owner authorization permits this
local engineering smoke; it does not establish G1 scientific sign-off or permit
redistribution. Raw data, trained weights and observation-derived samples stay
in ignored `outputs/`, never in Git.

Freeze the chronological training block before fitting normalization or windows:

- Full grid: 87,672 hourly UTC rows, hourly spacing checked.
- Original 80% cut: 70,137; training `[0,70041)`, 96-hour embargo before cut.
- Full threshold-event intervals checked for bounds, duplicate IDs and overlap;
  any interval crossing the purged training boundary is removed in full before
  windows. This is threshold-run containment, not meteorological storm validation.
- 70,041 finite training residuals; no imputation, no augmentation, no forcing.
- 69,946 stride-one windows, 72-hour history / 24-hour target, 69,969 unique target
  hours (1,678,704 overlapping forecast cells). Pilot differs deliberately from
  release 96/72 contract. No validation or test windows created or evaluated.
- 1,449 threshold-run IDs in the training block; full IDs in ignored data card.
  Independent storm count **unknown**. These runs are not independent storms.
- Mean 0.10825236834468384 m, SD 0.21461185616828657 m, fit on training rows only.
  Total-water q95/q99 recomputed on same block for provenance; unused by training.

`preparation.json` records exact hashes/counts and license limitations, without
raw observations. `outputs/issue6-prepare-verified/data_card.json` also contains
all threshold-run IDs. Upstream QC/tide processing is not independently certified;
this engineering smoke cannot repair or establish generalization boundaries.

## Implementation and commands

`torch_diffusion.py`: genuine conditional epsilon-predicting DDPM, cosine
cumulative schedule, sinusoidal diffusion-time embedding, history MLP consuming
all 72 past residuals, four dilated residual Conv1d blocks, 24-hour output.
`pilot_data.py`: fixed owner path, admission/migration/release checks, full-interval
purge and train-only scaling before window creation.

`train_gpu_pilot.py`: installed PyTorch only, explicit nonzero physical card,
CUDA visibility and UUID mapping checks, ownership evidence required, cooperative
per-card lock, repeated idle checks, foreign compute-process monitoring, 8 GiB
card-use/80 C abort limits and 30% PyTorch allocation cap. Batch <=128, <=3 epochs,
<=1800 seconds, BF16 when supported. Four BLAS/OpenMP environment variables and
PyTorch intra/inter-op threads are capped at 2; no DataLoader processes or pools.
The CUDA guard runs before optimizer creation; there is no CPU gradient path.

Every 20 steps, record physical GPU utilization/VRAM/temperature, process CPU time,
RSS, host load and PyTorch peak allocation; abort on foreign compute use.
Training produces config (SHA, source hashes, versions, seed, physical UUID),
per-step epsilon MSE, resources JSONL, checkpoint, eight generated samples,
conditional-sensitivity and nearest-training-window RMSE diagnostics. These are
implemented **but have not run on a GPU**. The near-copy check has only eight
queries; it cannot prove absence of memorization. Exception logs preserve the
last step; a conflict abort does not promise a usable partial checkpoint.

Actual preparation (no CPU gradients):

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 scripts/train_gpu_pilot.py \
  --prepare-only --output outputs/issue6-prepare-verified
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python3 -m pytest -q
ruff check src tests scripts/train_gpu_pilot.py
python3 -m compileall -q src scripts/train_gpu_pilot.py
git diff --check
```

The CPU gate covers scaling/window containment, entire boundary-event purge,
nonfinite/symlink refusal, temporal model forward shapes, finite outputs,
conditioning/timestep sensitivity, reproducible reverse sampling and mocked
foreign-process refusal. No backward/optimizer calls in the tests. PyTorch tests
skip when optional PyTorch is absent in base CI; NumPy integrity gates still run.
Installed validation environment: Python 3.10.12, NumPy 2.0.2, PyTorch 2.13.0+cu130.
See `pytest.txt` and `lint.txt` for final actual results. Initial lint findings were
corrected before the final checks. No packages/weights downloaded or installed.

### Pending launch after actual ownership evidence

Do not fabricate a reservation record. Save actual owner/scheduler confirmation
as JSON with `physical_index`, `unreserved: true`, `evidence_source`, `checked_at`
and a reference to the confirming message/record. Recheck that no other TideTail
trainer has started. The example below uses GPU4 only **if** subsequently confirmed
available; neither GPU4 nor GPU3 is currently approved by the available evidence.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
PYTHONPATH=src python3 -u scripts/train_gpu_pilot.py \
  --gpu 4 --reservation-record /tmp/tidetail-owner-confirmation.json \
  --epochs 3 --batch 128 --max-seconds 1200 \
  --output outputs/issue6-gpu4-run1 > /tmp/tidetail-gpu4-run1.log 2>&1
```

Retain stdout log alongside the output after the process exits. Commit source
before launch so the recorded SHA identifies it exactly. Save checkpoint and raw
samples locally; commit only aggregate run evidence. Examine the loss trajectory,
finite gradients, diagnostics and resource usage before calling the smoke complete.
This is 1,641 planned steps for three complete epochs, not an actual step count.
No run or publication claim is advanced here.

## Longer follow-up (not authorized by this engineering result alone)

1. Complete license/datum sign-off and frozen G1 event/time/station manifest with
   canonical storms and interval/embargo checks. Validate upstream preprocessing
   and freeze train/validation boundaries before selecting any model or threshold.
2. Use the same residual target, 72/24 or preregistered 96/72 contract, masks,
   forecast origins and training-only transforms for persistence, fixed seasonal,
   strong GRU/TCN and diffusion. Preserve per-event and unique-hour denominators.
3. Fix seeds, parameter counts, examples, optimizer steps, tuning budget and GPU
   time. Compare real-only, actual duplication/oversampling, perturbation, standard
   diffusion, then tail-aware controls with fixed augmentation fractions.
4. Evaluate held-out real storms only: overall/tail q95/q99 MAE/RMSE, peak magnitude
   and timing, detection precision/recall/false alarms, lead/station breakdowns,
   paired event bootstrap uncertainty; explicitly handle zero-event folds.
5. Assess ensemble calibration/coverage and proper scores, conditional sensitivity,
   spectra/duration/tail mass and nearest-neighbour memorization; downstream
   augmentation usefulness requires separate matched-budget forecaster tests.
   Use only issuance-available covariates. Keep all null/negative outcomes and
   link any later paper claim through `paper/CLAIMS_LEDGER.md`.
