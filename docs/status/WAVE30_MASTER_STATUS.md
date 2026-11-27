# Wave 3.0 Master Status — 2026-11-26 (updated 18:55 CST)

**Branch**: `trae/global-data-generative-research-20261031` · **PR**: #23 (Draft)
**Verified HEAD**: `f839384c78a7019f1e0dd1e68f4b8efa4ed62f02` (pushed)

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

- **W30-P2 RUNNING (verified 2026-11-26 18:51 CST)**: PID 3543561,
  `wave28_gen_bench_revised_eval.py`, 5 regions × 7 families × 2 gen seeds.
  CPU-only **by design** (CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=8; pilot
  fairness contract — NOT a CUDA fallback; no GPU process in nvidia-smi).
  Elapsed 1h00m at check, ~1015 s per family-seed, ETA ~20 h. Incremental
  atomic JSON writes verified (`wave30_generator_fidelity.json`).
  Sampling frame disclosure: strata select ~174 unique windows from the
  FULL 8,601-window test frame per region (NOT all 8,601 sampled).
  Charleston strata: 160 background (40/season) + 4 independent q95
  extremes + 10 q90 moderate-tail. Script has NO resume — process left
  untouched; safe-recovery implementation queued for post-run.
- **P1 COMPLETE**: five-region expanded eval + deterministic legacy
- **P1 COMPLETE**: five-region expanded eval done.
  - New regions: southeast_australia (all 8 augmented cells improve,
    −1.29…−3.02%, p ≤ 0.0005), thames_north_sea (all 8 harm, +3.44…+4.31%,
    p = 0.0001). NaN-window fix applied first (frozen window_index protocol).
  - Legacy 3 regions deterministically recomputed on GPU5 (drift ≤ 0.1 pp
    vs backup `wave28_expanded_eval_pre_recompute_20261126.json`); per-unit
    arrays + event IDs now persisted for ALL five regions.
  - Aggregate: 20/20 primary comparisons; 12 survive Bonferroni α=0.0025
    (6 beneficial / 6 harmful). Architecture pattern is DESCRIPTIVE, not
    universal: TCN responds more strongly than GRU in 4 of 5 regions, but
    Thames is an exception (GRU harm 4.30% > TCN 3.45%); direct paired
    interaction test DONE — see §4c.
    Artifacts: `wave30_expanded_five_region.json`,
    `wave30_expanded_event_metrics.csv` (1,441 rows),
    `docs/experiments/WAVE30_FIVE_REGION_EVAL.md`,
    paper Table `tab:five-region` (26-page PDF compiles clean).

## 4b. Headline scientific change vs Wave 2.9

The Wave 2.8 "no cell shows a seed-robust improvement" conclusion is now
SUPERSEDED by region-dependent results: generative augmentation produces
seed-robust BENEFIT in Southeast Australia and Gulf-Coast TCN, and
seed-robust HARM in Thames and Charleston TCN. Overall fidelity gate
status (INCONCLUSIVE everywhere) does not predict downstream value.
This reframes RQ1/RQ4: the question is no longer "does augmentation help"
but "which regions/architectures benefit and why" (P3/P4).

## 4c. Architecture interaction (DONE 2026-11-26)

Descriptive pattern (TCN responds more than GRU in 4/5 regions; Thames:
GRU +4.30% vs TCN +3.45% harm — GRU responds more) must NOT be stated as
a proven universal architecture effect.

Direct paired TCN−GRU interaction test on matched event units
(`scripts/wave30_arch_interaction.py` → `wave30_architecture_interaction.json`,
v2 corrected 2026-11-27):
10 paired tests (5 regions × std/tail arms), Bonferroni α=0.005.
**CORRECTION (Wave 3.2, verified independently): 3/10 survive, not 4/10.**
v1 compared the RAW signflip_p against α=0.005 (bug); the corrected v2
compares the Bonferroni-adjusted p. Survivors: US Gulf Coast −2.09 mm
(p_bonf=0.001), US Gulf tail −2.06 mm (p_bonf=0.001), SE Australia std
−1.29 mm (p_bonf=0.004). SE Australia tail −1.22 mm (p_bonf=0.008) does
NOT pass. Charleston REVERSES (+0.76 mm, ns after correction); Thames ns
(−2.41 mm, huge event variance); NY Bight ns. 8/10 interactions negative.
Conclusion unchanged: architecture interaction is region-dependent,
not universal — the paper states the 4/5 pattern as descriptive.

## 5. Standing constraints (unchanged)

PR #23 stays Draft; no main merge; no force-push; no deletion of history;
no raw datasets/weights/credentials in git; GPU0 prohibited; serial heavy
jobs; 679 = regional event units, not independent storms; `water_level_m`
is already non-tidal residual (never subtract tide again).
