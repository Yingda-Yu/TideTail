# Wave 3.0 Master Status — 2026-11-26

**Branch**: `trae/global-data-generative-research-20261031` · **PR**: #23 (Draft)
**Base head at wave start**: `0bb3663` (verified 2026-11-26 16:43 CST)

## 1. Server / process reconciliation (P0-01)

| Check | Finding | Action |
|---|---|---|
| TideTail processes | None running | — |
| systemd `tidetail-wave28physics.service` | state **failed** (stale; the 6-arm experiment actually completed 2026-11-20 out-of-band, JSON verified complete) | stale unit left untouched (history preserved); flagged in queue |
| GPU (2026-11-26 16:43) | GPU0 prohibited; GPU1/2/3/7 heavy; GPU4/6/9 light; **GPU5 & GPU8 ~free (286 MiB)** | Wave 3.0 GPU jobs use `CUDA_VISIBLE_DEVICES=5` (or 8) + `--device cuda:0` |
| RAM / disk | 503 Gi total, 179 Gi avail; /data 92% used, 658 Gi free | CPU-heavy jobs stay serial; no bulk downloads planned |
| Uncommitted residue | Wave 2.9 debug prints in `wave28_physics_fm.py`; import-order churn; 3 untracked scripts (`_cds_auth_test.py`, `_launch_gpu_jobs.sh`, `wave29_physics_resume.py` already used for the completed physics resume) | debug prints removed; residue committed with Wave 3.0 P0 |

## 2. Verified experiment state (P0-02)

| Artifact | State | Verification |
|---|---|---|
| `wave28_expanded_eval.json` (2026-11-20) | 3 regions complete: charleston, ny_bight, us_gulf_coast; A0 + standard + tail_aware, fm seeds 42/99, downstream 42/99/7; agg complete | parsed programmatically 2026-11-26 |
| `wave28_physics_fm.json` (2026-11-20) | both regions, all 6 arms × 2 fm seeds complete | parsed programmatically 2026-11-26 |
| `wave28_physics_analysis.json` | 20 primary comps, α=0.0025; p-value scan clean (no zero/NaN) | scanned 2026-11-26 |
| `wave28_gen_benchmark.json` (2026-11-08) | pilot, 2 regions × 7 families; contains 2 NaN diffusion_ts cells with spurious p=0 | corrected → `wave30_corrected_inference.json` |
| Five-region gap | **southeast_australia, thames_north_sea missing** from expanded eval | → W30-P1-01 |

## 3. Wave 3.0 P0 completions (2026-11-26)

1. **P0-03 zero-p fix**: `paired_event_stats` MC branch now `(b+1)/(n_boot+1)`;
   NaN/Inf/empty → explicit invalid; exchangeability documented; 8 regression
   scenarios added. Full suite 318 passed.
2. **P0-04 audit**: "11/28 significant harm" corrected to **9/26 valid
   survivors (all harmful, all REJECTED generators)**; 2 Diffusion-TS NY Bight
   cells reclassified generation failures. See
   `docs/audits/WAVE30_STATISTICAL_CORRECTNESS.md`.
3. **P0-05 resume**: `wave28_expanded_eval.py` gained `--resume` (cell-level
   completion checks, protocol-matched A0 reuse, atomic writes); benchmark +
   expanded scripts now persist `per_unit_mae` + event IDs.
4. Per-unit arrays + event IDs now persisted in every downstream cell.

## 4. Active work

- **W30-P1-01 (RUNNING next)**: southeast_australia + thames_north_sea
  expanded eval, protocol frozen identical to the 3 completed regions
  (`--epochs 15 --fm-seeds 42 99 --downstream-seeds 42 99 7 --n-aug 2000
  --n-test 600 --downstream-steps 200`), GPU5, resume-capable.

## 5. Standing constraints (unchanged)

PR #23 stays Draft; no main merge; no force-push; no deletion of history;
no raw datasets/weights/credentials in git; GPU0 prohibited; serial heavy
jobs; 679 = regional event units, not independent storms; `water_level_m`
is already non-tidal residual (never subtract tide again).
