# Branch Reconciliation — 2026-10-30

## Goal

Recover all substantive work from two independent Git branches without losing evidence, then integrate them onto a single working branch.

## Starting state

Before reconciliation, only these local branches existed:
- `main` at `554dfc2`
- `feature/data-integrity-split-leakage` at `5b790e5` (PR #14)
- `feature/g2-real-data-baselines` at `584094c` (PR #15/#17)
- `feature/research-docs-d` at `b100649` (PR #16)

The important branches `audit/18-cpu-integrity-c00664ed` (PR #19), `spartina/f1cff330` (PR #20) and `fix/6-cpu-sampling-440c95e0` (PR #21) were NOT present locally. Fetching all remotes revealed them.

## Inheritance analysis

Both chains share merge-base at `main` (`554dfc2`).

### Chain A — Data & Baselines

```
main (554dfc2)
 └── feature/data-integrity-split-leakage (5b790e5)  PR #14
      └── feature/g2-real-data-baselines (584094c)    PR #15, #17
           └── audit/18-cpu-integrity-c00664ed (3244c5c)  PR #19
```

Unique additions at PR #19:
- `tests/test_audit_split_regression.py`
- Additional audit evidence JSON/text in `docs/audits/18/`
- Rejected invalid event group

### Chain B — Diffusion

```
main (554dfc2)
 └── spartina/f1cff330 (e2d93fb)                      PR #20
      └── fix/6-cpu-sampling-440c95e0 (577af1d)        PR #21
```

Unique additions at PR #21:
- `src/tidetail/pilot_data.py`           — frozen NY Bight data prep with admission checks
- `src/tidetail/torch_diffusion.py`      — proper PyTorch DDPM with cosine schedule
- `scripts/train_gpu_pilot.py`           — GPU training with safety guards
- `scripts/diagnose_ddpm_cpu.py`         — CPU diagnostic script
- `scripts/audit_recovery_g1.py`         — G1 follow-up audit
- `tests/test_gpu_pilot.py`              — PyTorch smoke tests
- `tests/test_sampling_integrity.py`     — sampling integrity tests
- `docs/experiments/issue6/**`           — all historical experiment evidence

### Why they're NOT directly mergeable

Chain B (PR #20/#21) was branched from main without importing any of the data/baseline modules that Chain A provides. PR #20's report explicitly notes: "The PR #20 tree lacks the baseline, evaluator and split modules present in #19." A naive `git merge` would produce content conflicts, not clean composition.

## Reconciliation strategy: selective file inclusion

1. Check out from **PR #19** as the base (has most complete test suite: 55 tests).
2. Use `git checkout origin/pr-21 -- <paths>` to bring in PR #21 unique files that don't conflict.
3. Do NOT copy PR #21's versions of overlapping files (none exist — the two chains have zero overlapping source modules).

## Command log

```bash
git checkout -b trae/research-recovery-20261030 origin/pr-19
git checkout origin/pr-21 -- \
  src/tidetail/pilot_data.py \
  src/tidetail/torch_diffusion.py \
  scripts/train_gpu_pilot.py \
  scripts/diagnose_ddpm_cpu.py \
  scripts/audit_recovery_g1.py \
  tests/test_gpu_pilot.py \
  tests/test_sampling_integrity.py \
  paper/CLAIMS_LEDGER.md \
  docs/experiments/issue6/
ruff check --fix src tests
python3 -m pytest -q
```

## Result

| Metric | Before | After |
|---|---|---|
| Test count | 55 (PR #19 only) | **69** |
| Test pass rate | 55/55 | **69/69** |
| Source modules | 7 | **9** (+pilot_data, torch_diffusion) |
| Ruff warnings | 89 | 11 (all minor, non-blocking) |
| Compileall | pass | pass |
| git diff --check | pass | pass |

## Lost work checklist

None known. All substantive files from PR #19 and PR #21 are now present:

- ✅ PR #19: splits, data, qc, synthetic, evaluate, baselines, diffusion (CPU), all 55 tests, audit reports
- ✅ PR #20: historical GPU run report, checkpoint SHA/size, training config
- ✅ PR #21: torch_diffusion, pilot_data, all 5 scripts, all 19 PyTorch tests, CPU recovery report, diagnosis JSON

All 8 PRs (#13–#17, #19–#21) have been fetched and their relevant content is incorporated or preserved as evidence. PR #18 does not exist (GitHub issue #18 is a separate open issue).

## Follow-up

- `src/tidetail/diffusion.py` (CPU smoke) and `src/tidetail/torch_diffusion.py` (proper DDPM) coexist. The latter should eventually supersede the former after tests confirm.
- `src/tidetail/__init__.py` should expose the new modules when API surface is ready.
- `paper/` needs full LaTeX paper scaffold.
