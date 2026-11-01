# WORKLOG — TideTail

Format per entry: task ID, time, files modified, commands run, tests, commit, GPU, result.

---

## 2026-11-03

### Wave 1.7 — FM Retrain + A3 Pipeline + Paper P0 Caveats + CI Fix

| Field | Value |
|---|---|
| Task | P5 FM checkpoint retrain, wave17 A3 pipeline, paper P0 caveats, CI pandas fix |
| Branch | `trae/global-data-generative-research-20261031` (PR #23) |
| Head before | `9ce5b8f` (physics.py module) |
| Head after | (new commit — this work) |
| Start | 2026-11-03 21:27 UTC+8 |
| FM retrain | 21:27–21:47 — all 10 `flow_matching_{region}_seed{42,99}.pt` retrained on correct residual target (`residual = water_level_m`) |
| wave17 pipeline | PID 551931, GPU3, CUDA_VISIBLE_DEVICES=3, 22:21 start, ~70 min elapsed, 142/190 predictions |
| wave17_analyze.py fix | 3 bugs: (1) `ci['verdict']` KeyError at lines 185/199/214, (2) nonsense `.__class__` code at line 177, (3) inline verdict computation |
| Paper P0 caveats | 6-results.tex (fm benchmark section added critical caveat + retracted framework-choice attribution), 7-discussion.tex (rewrote "Why Flow Matching" section, updated limitations with current data status, updated augmentation section with wave17 results, updated "What this is" item 3) |
| CI fix | PR #23 failing: `ModuleNotFoundError: No module named 'pandas'`. Added `pandas>=2.0` + `scipy>=1.10` to `pyproject.toml` dependencies |
| LaTeX compile | ✅ 18 pages, bibtex resolved, all refs pass (3 passes: pdflatex + bibtex + pdflatex × 2) |
| Tests | ✅ 268 passed, 1 warning, 78s |
| Ruff | ✅ clean on all edited files |
| GPU used | GPU3 (wave17 pipeline, ~70 min so far) |
| Files staged | `scripts/wave17_analyze.py`, `pyproject.toml`, `paper/sections/6-results.tex`, `paper/sections/7-discussion.tex`, `docs/status/BLOCKERS.md`, `docs/status/MASTER_RESEARCH_STATUS.md`, `docs/status/WORKLOG.md`, `docs/status/EXPERIMENT_REGISTRY.csv`, `docs/status/DATASET_REGISTRY.csv`, `docs/status/FIGURE_REGISTRY.csv` |
| Result | **FM retrain COMPLETE** — blocker (b) RESOLVED. Wave17 pipeline executing with correct FM. Partial A3 results show region-dependent efficacy (ny_bight +15.5%, us_gulf +12.4%, thames -14.6% at ratio 0.5 GRU). Paper P0 caveats prevent misattribution of FM success to framework choice. CI pandas dependency added. |
| Next | Wait for PID 551931 completion (~20-30 min) → run wave17_analyze.py → update EXPERIMENT_REGISTRY → push to PR #23 |
| Status | **IN PROGRESS** (pipeline running, commit ready) |

---

## 2026-11-01

### Wave 1.6 P0/P2/P3 Completion + Status Registry Creation

| Field | Value |
|---|---|
| Task | Wave 1.6 closure + 7-file status registry |
| Branch | `trae/global-data-generative-research-20261031` (PR #23) |
| Head | `c6a703af4462e05bfc4a746b29d623e92e84e107` |
| Start | 2026-11-01 21:00 UTC+8 |
| Commands | git rev-parse HEAD, nvidia-smi, python3 -m pytest -q, cat CLAIMS_LEDGER.md, count JSON records |
| Tests verified | 218 passed, 1 warning, ruff clean |
| GPU used | None |
| Files added | `docs/status/MASTER_RESEARCH_STATUS.md`, `docs/status/BLOCKERS.md`, `docs/status/EXPERIMENT_REGISTRY.csv`, `docs/status/FIGURE_REGISTRY.csv`, `docs/status/DATASET_REGISTRY.csv` |
| Files updated | `docs/status/TASK_QUEUE.json` (version bump), `docs/status/WORKLOG.md` (this entry) |
| Result | **Wave 1.6 ACCEPTED**. P0 double-detiding fixed (fc45703), P2 total_water_eval rewritten (ad63c1b), P3 storm_events repaired (e88036f). All historical generative runs reclassified: A0 VALID (20 records), A3 EXPLORATORY (120 records), Diffusion-TS INVALIDATED (10 records), Flow Matching INVALIDATED (10 checkpoints). 7 status registry files created for PR #23. |
| Next | P5 G8 pilot retrain — verify GPU4 availability, retrain 10 FM checkpoints on correct residual target |
| Status | COMPLETE |

---

## 2026-10-30

### P0-1 Environment & Repo Audit

| Field | Value |
|---|---|
| Task | P0-1 |
| Branch | trae/research-recovery-20261030 (created from origin/pr-19) |
| Start | 2026-10-30 22:50 UTC+8 |
| Files modified | None (audit only) |
| Commands | pwd, git rev-parse, git status, git branch -avv, git log, nvidia-smi -L, nvidia-smi, python3 -c "import torch", pdflatex --version, latexmk -v |
| Tests run | 0 |
| Result | Repo clean, remote correct, 10× RTX3090 (GPU 0/3/4 free), Python 3.10.12 + torch 2.13.0+cu130, TeX Live 2022 functional |
| GPU used | None |
| Status | COMPLETE |

### P0-2 Branch Reconciliation

| Field | Value |
|---|---|
| Task | P0-2 |
| Start | 2026-10-30 22:55 UTC+8 |
| Files added | src/tidetail/pilot_data.py, src/tidetail/torch_diffusion.py, scripts/{train_gpu_pilot,diagnose_ddpm_cpu,audit_recovery_g1}.py, tests/{test_gpu_pilot,test_sampling_integrity}.py, docs/experiments/issue6/**, paper/CLAIMS_LEDGER.md (overwrite), docs/audits/TRAE_FULL_AUDIT.md, docs/audits/BRANCH_RECONCILIATION.md, docs/plans/RESEARCH_EXECUTION_PLAN.md, docs/status/WORKLOG.md |
| Commands | git remote update, git checkout -b trae/research-recovery-20261030 origin/pr-19, git checkout origin/pr-21 -- ..., ruff check --fix, python3 -m pytest -q |
| Tests run | 55 (PR #19 base) + cherry-pick PR #21 files → **69 tests pass** |
| Result | Two independent branch chains reconciled via selective file inclusion. Zero overlapping source modules. 82 ruff warnings auto-fixed. |
| GPU used | None |
| Status | COMPLETE |

### P0-3 Data Directory Verification

| Field | Value |
|---|---|
| Task | P0-3 |
| Data path | `/data/yingda/tidetail-data/prime-formal-2015-2024` |
| Size | 451 MB, 350 files |
| Checkpoint path | `/data/yingda/spartina-agent/worktrees/spartina/f1cff330/outputs/issue6-gpu4-pilot-7164/checkpoint.pt` |
| Checkpoint size | 444,971 bytes |
| Migration manifest | EXISTS at `/data/yingda/tidetail-data/MIGRATION_MANIFEST.json` (supersedes PR #19 missing finding) |
| Status | COMPLETE (physical verification, SHA not re-verified) |

### P0-4 LaTeX Compile Test

| Field | Value |
|---|---|
| Task | P0-4 |
| Commands | cat → /tmp/min_latex/test.tex, latexmk -pdf -halt-on-error test.tex |
| Result | 85,963-byte PDF produced, zero errors |
| Status | COMPLETE |

---

## Next Tasks (Ordered by Dependency)

| Task | Stage | Depends On | Priority | GPU |
|---|---|---|---|---|
| G8 Pilot FM retrain | S5 | Wave 1.6 | HIGH | YES (GPU4) |
| A3 rerun (correct targets) | S5 | G8 retrain | HIGH | YES |
| FM failure diagnosis | S5 | A3 rerun | HIGH | No (CPU) |
| ERA5 + GEBCO integration | S4 | CDS auth | MEDIUM | No |
| Tail-aware FM | S6 | G8 retrain | HIGH | YES |
| Physics-guided priors | S6 | Tail-aware FM | MEDIUM | YES |
| Competing generative benchmarks | S6 | P5 | HIGH | YES |
| Generalization evaluation | S7 | Benchmarks | MEDIUM | No |
| Publication figures | S7 | Results | MEDIUM | No |
| Manuscript revision | S7 | Figures | HIGH | No |
| Reproducibility package | S7 | Manuscript | HIGH | No |
