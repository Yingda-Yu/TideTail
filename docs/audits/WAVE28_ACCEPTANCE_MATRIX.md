# Wave 2.8 Scientific Acceptance Matrix

**Compiled**: 2026-11-08 23:15 CST (draft; physics-dependent items updated on completion)
**Branch**: `trae/global-data-generative-research-20261031` · **PR**: #23 (DRAFT, do not merge)
**Head**: `373a5bc` (pushed)

Status legend: PASS (evidence complete) / FAIL (defect, repaired or reported) /
INCONCLUSIVE (no claim made) / BLOCKED-PENDING (waiting on the running physics
unit) / NOT-CLAIMED (explicitly not claimed in manuscript).

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Data integrity & ERA5 readiness | **PASS** | `data/results/era5_readiness_matrix.json`: 600/600 partitions, 87,672 h/region, 119/119 station continuity, `all_regions_integrity_ok: true`; target/tide merge audit in `wave27_event_set_audit.json` |
| 2 | Event independence & census arithmetic | **PASS** | Union 679 (634 POT + 73 TC − 28 matched); 2024: 59 units; 0 duplicate timestamps; frozen strict-overlap matching + tolerance sensitivity (676/667/665); `tests/test_event_set_arithmetic.py` |
| 3 | E26 sampling-defect fix (causality + correctness) | **PASS** | Commit `9b790a6`; regression tests `tests/test_wave28_physics.py`, `tests/test_physics_causality.py` (5 tests); code SHA `1106ecbb` recorded per-run in `env.txt`; verified at full scale: wave27 physics 10/10 cells INCONCLUSIVE; expanded rerun scale 1.147/1.097 |
| 4 | Generative fidelity gates (Wave 2.7 5-variant) | **INCONCLUSIVE** (honest) | `data/results/wave27_physics_fm.json` (sha256 `53f5752…`, exit 0): all 10 variant×region cells fidelity-INCONCLUSIVE (scale 0.869–1.077) — gate = quality.py hard window [0.5, 2.0] + warnings; no cell ACCEPTED; correctly reported as viable-not-accepted |
| 5 | Seven-family benchmark (pilot) | **PASS** (as pilot) | `data/results/wave28_gen_benchmark.json` (sha256 `3656e1ed…`); FM/tail-FM/WGAN-GP INCONCLUSIVE both regions; cvae/timegrad/ddpm/dts REJECTED; P0-4 pilot contract in `docs/experiments/WAVE28_GEN_BENCHMARK.md`; family-ranking claims absent; 28-comparison inference corrected: 11 survive Bonferroni α=0.0018, all harms of REJECTED generators |
| 6 | Benchmark fidelity sample audit (P3) | **CAVEATED + REVISED EVAL QUEUED** | Pilot scored first 200 consecutive test windows (single early-winter slice) — documented in doc/manuscript/ledger; revised season-stratified evaluation scripted (`scripts/wave28_gen_bench_revised_eval.py`, strata verified: 4 seasonal×background 40 + q95/168h independent extremes + q90/120h moderate tail, 2 gen seeds, per-stratum tail95/tail99) and chained via systemd unit `tidetail-wave28bench-rev` to start only after physics succeeds; pilot artifact preserved unchanged |
| 7 | Expanded-year evaluation (leakage-controlled) | **PASS** (honest negative) | `data/results/wave28_expanded_eval.json`: fidelity restored post-fix; standard-FM augmentation **harms** extreme-event MAE (+0.86% p=0.016 / +0.92% p=0.0023); 6-cell comparison: no seed-robust improvement; independently verified; exploratory framing in manuscript |
| 8 | Physics-prior 6-arm ablation (W28) | **BLOCKED-PENDING** | systemd unit `tidetail-wave28physics` (run `wave28physics_20261108T133917Z_39437029`, relaunched 21:39 after oomd kill under 3-way concurrency); incremental saves verified; partial log shows healthy fidelity (INCONCLUSIVE scales 0.897/1.009); pre-specified contrast analysis scripted (`scripts/wave28_physics_analysis.py`: 5 mechanism contrasts × 2 regions × 2 archs = 20 primary, Bonferroni α=0.0025; A0-secondary exploratory) |
| 9 | Statistical inference & multiplicity | **PASS** | Wave 2.7: 320-test family → 0 seed-robust survivors (bootstrap p-value repaired, MC resolution floor); wave27 physics: nothing survives α=0.005; benchmark: correct Bonferroni family (28) with survivors honestly reported as rejected-generator harms; sign-flip null on per-unit paired errors throughout |
| 10 | Reproducibility & code hashes | **PASS** | Immutable run dirs + status/env/checksums via `scripts/tidetail_run.sh`; append-only `data/results/RUN_MANIFEST.jsonl`; per-run code SHA `1106ecbb`; seeds logged (FM 42/99, downstream 42/99/7, benchmark gen 42); no raw datasets or checkpoints in git |
| 11 | Manuscript completeness | **PASS with 1 pending ref** | 6-results.tex: wave27 corrected-run §6.6 + seven-family pilot §6.3b (interpretation contract, training-budget table) + six-arm skeleton with pre-specified contrasts; discussion/conclusion/limitations updated (pilot budgets, fidelity-sample caveat, seed-coverage limitation); pending: `tab:physics-ablation` numbers + figure regeneration after unit completes |
| 12 | Figure correctness | **PASS (2/3)** | `W28_benchmark.pdf` + `W28_expanded_cells.pdf` regenerated with correct [0.5, 2.0] gate (log axis); `W28_physics_ablation.pdf` is interim-partial — regenerate on completion |
| 13 | LaTeX PDF compilation | **PASS** | latexmk exit 0; single unresolved reference = pending physics table (expected); visual inspection pending final numbers |
| 14 | Test suite | **PASS** | `pytest -q` CPU: 311 passed (2026-11-08 23:05) |
| 15 | Claim traceability | **PASS** | `paper/CLAIMS_LEDGER.md` E24 (wave27 physics corrected), E28 (frozen matching), E30 (benchmark pilot, corrected statement) complete; E29 (6-arm) pending artifact; every numerical statement in the pilot subsection checked against `wave28_gen_benchmark.json` |
| 16 | Simulated independent peer review | **PENDING** (final pass after physics) | First-pass review already applied: caught ±10% gate error (P1), false "none survives Bonferroni" (P2), single-season fidelity sample (P3), mis-assumed comparison count (P4) — all corrected this session |
| 17 | Physics-scope honesty | **PASS** | Manuscript + docs + ledger state: ERA5 arms = meteorological conditioning (retrospective reanalysis); prior arms = physics-inspired statistical regularization (train-only OLS, auto-calibrated frozen λ); NO PDE constraints; NO bathymetry (GEBCO deferred); no operational-forecast claim |
| 18 | Execution reliability (P0-1) | **PASS** | systemd user units survived context resets (physics unit alive across session switch); oomd kill root-caused (3-way concurrency) and policy set: never >1 heavy job; chained waiting unit demonstrates session-independent orchestration |
| 19 | Artifact protection (P0-2) | **PASS** | Immutable per-run dirs, backup-before-overwrite, atomic writes; benchmark merge preserved pre-merge backup; revised eval writes separate artifact; Charleston pilot artifact preserved verbatim |

**Outstanding before final sign-off**: items 8 (physics ablation), 12 (ablation
figure), 13 (final visual PDF inspect), 16 (post-physics peer review), then
final commit/push. Physics ETA ~03:30 CST; revised fidelity eval auto-starts
on physics success.
