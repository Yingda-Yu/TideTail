# Issue #6: actual owner-authorized CUDA pilot, 2026-10-25

Continues draft PR #20 and `spartina/f1cff330`; no duplicate PR or merge.
Owner authorization: task d26eff7c, supplied `/data/tmp/yingda/spartina-prompt-7164a6b6.txt`.
This supersedes the allocation blocker in the preserved [preparation report](REPORT.md).

## Outcome and limitations

**Real GPU optimization succeeded: 1,641 total steps, three complete epochs,
209,838 overlapping training examples seen.** Five smoke steps were resumed for
1,636 further steps, without replaying those first 640 examples. AdamW state in
the final reloaded checkpoint has step 1,641 for all parameters. All losses and
checked gradient norms were finite. This is engineering success only.

**Sample quality is negative:** eight finite 24-hour residual sequences span
-16.910 to +7.940 m, with SD 4.474 m (training normalization SD 0.214612 m).
These amplitudes are implausible for this coastal residual pilot. Declining
training epsilon MSE does not establish convergence, physical realism or forecast
skill. Do not use these samples for augmentation. Independent storm count remains
unknown; threshold runs and overlapping windows are not independent samples.
No validation/test evaluation, model selection or publication claim was made.

| Quantity | Actual result |
|---|---|
| Model | 27,169-parameter history-conditional temporal epsilon DDPM, 64 diffusion steps |
| Training source SHA | `5bc646676a08191a77e5d424f2cf52bb1af00c04`, clean working tree at both successful launches |
| Physical GPU | RTX3090 index 4; UUID `GPU-6be030f8-625e-c4e7-a2fb-1309a673664b` |
| CUDA mapping | `CUDA_VISIBLE_DEVICES=4`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`; only logical `cuda:0`; UUID matched |
| Pilot timestamps UTC | 2026-10-25 13:00:36.023167 to 13:01:17.505885 |
| Smoke / resumed duration | 2.373 s / 41.477 s, training-and-diagnostics timer; startup/data loading excluded |
| Epoch mean epsilon MSE | 0.219863, 0.180404, 0.172029 |
| First / final step MSE | 1.100233 / 0.150355 |
| First / last 100-step mean MSE | 0.311275 / 0.170703 |
| Observed pilot resources | 84 snapshots; GPU utilization 0–19%, max VRAM 428 MiB, peak tensor allocation 52.947 MiB, max 34 C |
| Host resources | Mean process CPU 45.28% of one core; max RSS 1,918,132 KiB; host load ~19–20 on a 96-logical-CPU host |
| Limits | Four BLAS/OpenMP caps 2; torch intra/inter-op 2; workers 0; batch 128; BF16; AdamW lr 0.001; seed 20261025 |
| Budget | Three epochs, 1,200-second cap; completed all epochs without reaching timebox |

Live host checks repeatedly found GPUs 3/4 idle: GPU4 18 MiB, 0%, 25–26 C and
no compute process before launch. Other cards had owner/collaborator workloads;
GPU0 was excluded. `squeue`/`sinfo`, scheduler processes, `/etc/slurm*`, `/run/lock`
and relevant `/tmp` reservation/lock names revealed no GPU reservation mechanism.
This is absence of reservation indicators, not an invented scheduler release.
The explicit owner instruction permits selecting a live free card; its text and
these checks are the allocation evidence. No additional confirmation was required.
No sudo, process termination, shared-setting changes, installs or downloads.
The cooperative card lock and resource/foreign-process guard ran throughout;
no conflict was detected. GPU4 was idle again after both training and reload.

## Data protocol

Read only the authorized owner migration at
`/data/yingda/tidetail-data/prime-formal-2015-2024` and its parent migration manifest;
no collaborator originals. Migration, release and formal-admission hashes passed.
Only admitted ny_bight, station `NOAA_8518750_The_Battery`; HK excluded.
The original chronology is frozen and full threshold intervals touching the
boundary are purged before train-only scaling/windowing. Training rows `[0,70041)`,
original cut 70137, 96h embargo; 69,946 stride-one 72h→24h windows,
69,969 unique target hours; 1,449 threshold-run IDs retained in local data cards.
No augmentation, imputation or future covariates. No validation/test windows.
Normalization mean 0.1082523683 m, SD 0.2146118562 m, training rows only.
Full event IDs, station IDs and source hashes are in each local `data_card.json`.

This checks threshold-interval containment, not canonical storm independence or
upstream tide/QC leakage. License/redistribution, gauge datum, frozen G1 storm
splits and G2 baselines remain unsigned. Local owner engineering authorization
does not establish scientific sign-off or redistribution rights.

## Defects, checkpoint and reload evidence

The first attempted smoke stopped before optimizer creation: installed PyTorch
returns a bare UUID while nvidia-smi prefixes it with `GPU-`. The corrected guard
parses both as UUIDs and still rejects a different or invalid UUID; CPU regression
covers this. The failed attempt and log remain local, without any optimizer steps.
An initial commit command also failed for missing host Git identity; source was
subsequently committed as the existing `Codex <codex@openai.com>` identity before
either successful launch. No failed attempt is counted as training.

Added `--max-steps`, trusted-local `--resume`, source/data/config compatibility
checks, deterministic epoch permutations and next-offset/RNG/optimizer state.
Checkpoints are written atomically every 100 steps, at success, and on caught
training exceptions. Abrupt process termination can lose work since the last save.
The smoke's historical `timebox_reached: true` field means the combined budget
check stopped it; it actually hit the five-step cap, not the 60-second limit.

Final checkpoint (ignored, not uploaded):
`/data/yingda/spartina-agent/worktrees/spartina/f1cff330/outputs/issue6-gpu4-pilot-7164/checkpoint.pt`

SHA256: `dee516cf74b50a0ca8aff8ab05815c7ed36a32ea6c49b9dc81d5b532d39ebb68`.
A separate CUDA process reloaded model/optimizer state and confirmed `(8,1,24)`
finite output, exact repeatability with seed 99, and equality with saved samples.
A +1 standardized-history shift changed mean absolute output by 30.84645
standardized units: conditioning is active, but such sensitivity is not realism.
Eight nearest-training-window RMSEs span 0.16977–41.74372 standardized units;
none below 1e-4. Eight queries cannot establish absence of memorization.

All paths below are relative to the existing `f1cff330` worktree:

- `outputs/issue6-gpu4-pilot-7164/`: checkpoint, per-step `loss.csv`, `samples.npz`,
  config/SHA/source hashes, data card, resource JSONL, summary, aggregate, reload check.
- `outputs/issue6-gpu4-smoke-7164b/`: successful smoke checkpoint and all run evidence.
- `outputs/issue6-gpu4-smoke-7164/`: failed pre-optimizer UUID-check preparation.
- `outputs/issue6-owner-7164/`: allocation record, hardware snapshot, `smoke.log`,
  `smoke-b.log`, `pilot.log`, `reload.log`, exact `verify_checkpoint.py`, Python/package
  versions, `pip-freeze.txt`, and final hardware/ownership snapshot.

Only aggregate evidence is committed: `gpu_7164_{aggregate,summary,reload}.json`,
hardware text and CPU gate logs beside this report. Raw observations, samples,
checkpoints, optimizer states and full local logs remain ignored.

## Exact executed commands

Commands ran on the host from the existing branch/worktree, with no CPU training:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src
python3 -u scripts/train_gpu_pilot.py --gpu 4 \
  --reservation-record outputs/issue6-owner-7164/allocation.json \
  --epochs 3 --batch 128 --max-steps 5 --max-seconds 60 \
  --output outputs/issue6-gpu4-smoke-7164b > outputs/issue6-owner-7164/smoke-b.log 2>&1
python3 -u scripts/train_gpu_pilot.py --gpu 4 \
  --reservation-record outputs/issue6-owner-7164/allocation.json \
  --resume outputs/issue6-gpu4-smoke-7164b/checkpoint.pt \
  --epochs 3 --batch 128 --max-seconds 1200 \
  --output outputs/issue6-gpu4-pilot-7164 > outputs/issue6-owner-7164/pilot.log 2>&1
python3 outputs/issue6-owner-7164/verify_checkpoint.py > outputs/issue6-owner-7164/reload.log 2>&1
CUDA_VISIBLE_DEVICES='' python3 -m pytest -q
ruff check src tests scripts/train_gpu_pilot.py
python3 -m compileall -q src scripts/train_gpu_pilot.py
git diff --check
```

CPU gate: 11 passed (see `gpu_7164_pytest.txt`); lint and compile passed.
Host environment: Python 3.10.12, NumPy 2.2.6, PyTorch 2.13.0+cu130.
Earlier report's NumPy version describes its earlier environment, not this run.

## Prepared longer controlled command and next research

Not launched. Recheck free-card/ownership conditions and refresh the allocation
record first; never reuse a stale observation as availability evidence. Keep
monitoring and caps. A new engineering run with smaller batches allows more
optimizer updates within the existing three-epoch/30-minute guard:

```bash
# With the same exports and a freshly verified GPU4 allocation record:
python3 -u scripts/train_gpu_pilot.py --gpu 4 \
  --reservation-record outputs/issue6-owner-7164/allocation.json \
  --epochs 3 --batch 16 --max-seconds 1800 \
  --output outputs/issue6-gpu4-longer-NEXT > /tmp/tidetail-longer-NEXT.log 2>&1
```

This is a fresh run, not continuation from the completed three-epoch checkpoint:
`--resume --epochs 3` on that checkpoint would add zero updates. Changing batch
size changes the optimization budget (13,116 planned steps), so this is not a
matched-budget comparison. A larger epoch cap needs a separately reviewed change.
Before spending more budget, investigate the reverse sampler's terminal-step
sensitivity and train-only fixed-noise denoising/calibration diagnostics; do not
select fixes against validation/test outcomes. Preserve the current poor samples.

Complete canonical event/time/station G1 splits and license/datum sign-off, then
freeze comparable persistence/GRU/TCN and A0–A3 budgets. Add ERA5 only when its
forecast vintage and availability at issuance are verified; future observed ERA5
must be a separately labelled oracle, never operational conditioning. Precommit
multiple seeds, training/tuning budgets and augmentation fractions. Evaluate real
held-out storms with q95/q99 train-fixed tail MAE/RMSE, peak amplitude/timing,
threshold precision/recall/false alarms, lead/station breakdown and paired
event-level bootstrap intervals, retaining null/negative results. Trace any later
paper claim through the claims ledger. Issue #6 stays open; PR #20 stays draft.
