# Self-Hosted CI Audit — Wave 3.0

## Latest verified run — 2026-11-26 18:55–18:56 CST

| Field | Value |
|---|---|
| Command | `PYTHONPATH=src python3 -m pytest -q` |
| Commit SHA | `d0b577ac4196392b7754d46ef8dd6e2ab427b8b0` (pushed) |
| Start / finish | 2026-11-26T18:55:09+08:00 / 18:56:18+08:00 (67.73 s) |
| Exit code | 0 |
| Result | **320 passed, 0 failed, 0 skipped, 1 warning** |
| ruff | `ruff check src tests` — All checks passed (exit 0) |
| `git diff --check` | exit 2 — trailing whitespace in `paper/main.log` only (LaTeX build artifact, regenerated per compile; no source-file issues) |
| Warning | torch requires_grad→scalar conversion (test_baselines.py), benign |
| Full log | `data/logs/ci_run_20261126_p5.log` |

### Count reconciliation (P5 requirement)

| Historical record | Count | Status |
|---|---|---|
| Self-hosted CI audit (Wave 2.9, 2026-11-20) | 311 passed | valid for commit `373a5bc`-era tree only — superseded |
| Wave 3.0 interim queue notes | 318 passed | valid at the P0-fix tree (2026-11-26 ~17:00) — superseded |
| **This run (HEAD `d0b577a`)** | **320 passed** | **current evidence** |

Delta 311→318: +8 `paired_event_stats` regression tests, −1 obsolete
sample_windows stub incompatibility fixed. Delta 318→320: +2
`build_expanded_splits` NaN-exclusion tests (`tests/test_expanded_splits_nan.py`).

**Rule reaffirmed**: test counts are only valid for the exact commit they
were run against; older counts must not be cited as current CI evidence.

---

## Historical: Wave 2.9 audit (2026-11-20) — preserved below

## Runner
- **Host**: self-hosted Wave 2.9 CI server (not GitHub Actions)
- **Reason**: User migrated CI; GitHub Actions status is legacy
- **Python**: 3.10.x
- **GPU**: RTX 3090 × 10 (GPU0 PROHIBITED by llama-server + starvla)

## Check Matrix

| Check | Command | Result | Notes |
|-------|---------|--------|-------|
| **ruff src** | `ruff check src` | ✅ ALL PASS | Wave 2.8 残留的 wgan_gp.py unused import 已自动修复 |
| **ruff tests** | `ruff check tests` | ✅ ALL PASS | auto-fixed |
| **ruff src tests** | `ruff check src tests` | ✅ ALL PASS | zero errors |
| **pytest** | `pytest -q` | ✅ **311/311 PASS** | 1 warning (torch requires_grad → scalar conversion in test_baselines.py) |
| **pytest runtime** | — | **76.38s** | no OOM, no timeout |
| **GPU training** | `train_variant cuda:0` | ✅ 22.5s/10 epochs | GPU3 RTX 3090 |
| **GPU sampling** | `sample_windows cuda:0` | ✅ 0.14s/draw | |
| **torch threads** | `torch.set_num_threads(4)` | ✅ | capped per Wave 2.8 OOM lesson |

## pytest Details

```
311 passed, 1 warning in 76.38s

Warning:
tests/test_baselines.py::test_gru_training_decreases_loss
  UserWarning: Converting a tensor with requires_grad=True to a scalar
  may lead to unexpected behavior. Consider using tensor.detach() first.
  (Triggered internally at /torch/csrc/autograd/generated/python_variable_methods.cpp)
```

This warning is benign — test reads `float(model.forward_loss(...))` on a tensor
with `requires_grad=True`. No functional impact.

## Wave 2.8 OOM Lessons Applied

| Prevention | Implementation | Verify |
|------------|---------------|--------|
| ≤8 CPU threads per process | `torch.set_num_threads(min(8, ...))` | ✅ |
| One heavy CPU task at a time | Physics FM downstream eval serial in main() | ✅ |
| GPU resource isolation | `CUDA_VISIBLE_DEVICES=3` (not 0) | ✅ |
| OOM-safe launch | `setsid ... & disown`, stdin/stdout detached | ✅ |
| Resume capability | `--resume` flag per Wave 2.9 patch | ✅ |
| Incremental checkpoint | `out_path.write_text(...)` per arm, not per region | ✅ |

## Environment State

```bash
# GPU inventory 2026-11-20
GPU  Free    Used    Users
0    24GB    22MB    llama-server PROHIBITED
1    1.4GB   22.7GB  starvla-policy × 2
2    2.4GB   21.7GB  comfy-h3
3    25.0GB  24MB    Physics Resume (THIS RUN)
4    6.2GB   17.9GB  llama-server-qwen35
5-9  2-22GB  —       other users

# CDS API
~/.cdsapirc exists, cdsapi 0.7.7, NO token leak

# Git
Branch: trae/global-data-generative-research-20261031
PR: #23 (DRAFT, DO NOT MERGE main)
Git HEAD: 373a5bc (2026-11-08 prior session)
```

## Systemd Services

All Wave 2.8 transient units dead — **no persistent systemd config**.
Wave 2.9 uses `setsid bash -c '...' < /dev/null & disown`.

## Success Criteria

Wave 2.9 passes all three gates:
1. ✅ **ruff src tests** — zero errors
2. ✅ **pytest -q** — 311/311 pass
3. ✅ **GPU training** — FM train + sample on cuda:0 without device mismatches

## No Fabrication Confirmation

Wave 2.8 charleston 4-arm physics result stands as-is — **NO fabrication**:
- Every completed arm's Δ% < 1.5%
- All signflip_p > Bonferroni α=0.0025
- Negative result is honest and reportable
- Wave 2.9 charleston combined_prior + combined_tail will either confirm or
  contradict; if also negative, conclusion strengthens
- Wave 2.9 ny_bight is the critical test — if physics shows benefit there,
  charleston may be an outlier region-specific finding
