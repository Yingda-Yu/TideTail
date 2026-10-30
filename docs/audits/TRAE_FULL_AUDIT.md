# TRAE Full Audit — 2026-10-30

Branch: `trae/research-recovery-20261030`
Base: `origin/pr-19` (`3244c5c`) + cherry-pick of `origin/pr-21` unique files
Auditor: TRAE Agent

Status legend: **VERIFIED / PARTIAL / FAILED / BLOCKED / NOT RUN**

---

## 1. Repository and Environment

| Item | Status | Evidence |
|---|---|---|
| Working tree clean | VERIFIED | `git status --short` → empty on start of branch |
| Remote origin | VERIFIED | `https://github.com/Yingda-Yu/TideTail.git` |
| Python | VERIFIED | 3.10.12 |
| PyTorch | VERIFIED | 2.13.0+cu130, CUDA 13.0 |
| NumPy | VERIFIED | 2.2.6 |
| pytest | VERIFIED | Installed at `~/.local/bin/pytest` |
| ruff | VERIFIED | Installed at `~/.local/bin/ruff` |
| nvidia-smi | VERIFIED | Driver 580.159.03, CUDA 13.0 |
| TeX Live | VERIFIED | latexmk 4.76, pdflatex, xelatex, kpsewhich, bibtex |
| biber | BLOCKED | Not installed; bibtex is sufficient for now |
| Article.cls | VERIFIED | `/usr/share/texlive/texmf-dist/tex/latex/base/article.cls` |
| Minimal LaTeX compile | VERIFIED | `/tmp/min_latex/test.pdf` 85,963 bytes |

## 2. GPU Status — 10× RTX 3090

| GPU | UUID | Memory | Util | Status |
|---|---|---|---|---|
| 0 | GPU-b26ce733-… | 22/24576 MiB | 0% | **FREE** |
| 1 | GPU-b007c532-… | 18549/24576 MiB | 0% | BUSY (llama-server, python) |
| 2 | GPU-5b231897-… | 22255/24576 MiB | 0% | BUSY (comfy-h3) |
| 3 | GPU-9cd80b3c-… | 24/24576 MiB | 0% | **FREE** |
| 4 | GPU-6be030f8-… | 18/24576 MiB | 0% | **FREE** (historical pilot GPU) |
| 5 | GPU-9df8f39a-… | 15176/24576 MiB | 100% | BUSY (multiple python) |
| 6 | GPU-3035ddcc-… | 16866/24576 MiB | 100% | BUSY (renderorbit, multiple python) |
| 7 | GPU-25c090c7-… | 17803/24576 MiB | 100% | BUSY (multiple python) |
| 8 | GPU-89634698-… | 16113/24576 MiB | 100% | BUSY (multiple python) |
| 9 | GPU-36f8dc5d-… | 15429/24576 MiB | 100% | BUSY (multiple python) |

**Conclusion**: GPUs 0, 3, 4 are free. Must not assume GPU 0 is unreserved.

## 3. Data Directory

| Item | Status | Evidence |
|---|---|---|
| Data path exists | VERIFIED | `/data/yingda/tidetail-data/prime-formal-2015-2024` |
| File count | VERIFIED | 350 files |
| Total size | VERIFIED | 451 MB |
| File inventory SHA | VERIFIED | `file_inventory_sha256.json` present |
| Migration manifest | VERIFIED | `/data/yingda/tidetail-data/MIGRATION_MANIFEST.json` exists (supersedes PR #19 "missing" finding) |
| Region admission | PARTIAL | Only ny_bight admitted; HK formally excluded |
| Per-station license/datum | BLOCKED | Not yet signed off by data owner |

## 4. Historical Checkpoint

| Item | Status | Evidence |
|---|---|---|
| Path exists | VERIFIED | `/data/yingda/spartina-agent/worktrees/spartina/f1cff330/outputs/issue6-gpu4-pilot-7164/checkpoint.pt` |
| File size | VERIFIED | 444,971 bytes |
| SHA256 | NOT RUN | Not re-verified in this audit |
| Architecture | VERIFIED | 27,169-parameter temporal epsilon DDPM (PR #20 report) |

## 5. Branch Merge — Completed

**Base**: `origin/pr-19` (`3244c5c`) — audit: verify Prime data and CPU baselines; reject invalid event groups
**Additions from**: `origin/pr-21` (`577af1d`) — fix: CPU sampling integrity, torch_diffusion, pilot_data, GPU training script

### Two independent inheritance chains from main (`554dfc2`):

1. **Data/Baseline chain** (merged as base):
   - main → `feature/data-integrity-split-leakage` → `feature/g2-real-data-baselines` → `audit/18-cpu-integrity-c00664ed`
   - Contains: splits, data, qc, synthetic, evaluate, baselines, diffusion (CPU smoke), audit regression tests

2. **Diffusion chain** (cherry-picked unique files):
   - main → `spartina/f1cff330` → `fix/6-cpu-sampling-440c95e0`
   - Added: `pilot_data.py`, `torch_diffusion.py`, `train_gpu_pilot.py`, `diagnose_ddpm_cpu.py`, `audit_recovery_g1.py`, `test_gpu_pilot.py`, `test_sampling_integrity.py`, all experiment evidence docs

### Test results

| Branch | SHA | Tests | Result |
|---|---|---|---|
| feature/g2-real-data-baselines | 584094c | 50 | 50 passed |
| PR #19 | 3244c5c | 55 | 55 passed |
| PR #21 | 577af1d | 19 | 19 passed |
| **trae/research-recovery-20261030 (merged)** | 3244c5c + pr-21 files | **69** | **69 passed** |

## 6. Historical Experiment Findings (from reports, not rerun)

### PR #20 GPU Pilot (1,641 optimizer steps, 3 epochs on RTX3090 GPU4)

| Metric | Value |
|---|---|
| Architecture | 27,169-param temporal epsilon DDPM, 64 diffusion steps |
| Training loss (epoch means) | 0.219863 → 0.180404 → 0.172029 |
| Sample range | **−16.910 to +7.940 m** |
| Sample SD | 4.474 m |
| Training normalization SD | 0.215 m |
| **SD ratio** | **22.95×** |
| Terminal alpha_bar | 5.93e-7 |
| Terminal epsilon→clean amplification | 1298.8× |

**Verdict**: NEGATIVE — sample amplitudes implausible, training loss decline does not establish convergence or physical realism.

### PR #21 CPU Independent Reproduction

| Result | CUDA artifact | CPU seed 99 |
|---|---|---|
| Minimum residual m | −16.910 | −13.079 |
| Maximum residual m | 7.940 | 16.277 |
| Sample SD m | 4.474 | 4.922 |

**Verdict**: Confirms magnitude failure. No fundamental equation bug found. Root cause likely: terminal step sensitivity + imperfect epsilon prediction + training budget.

### PR #21 Engineering Guards (added, model NOT fixed)

- Nonfinite rejection for conditioning, epsilon, reverse states
- Train-reference scale screen (SD ratio [0.1, 5.0], mean offset ≤ 5 train SD)
- **These are validation guards, not model repairs.**

## 7. Remaining Ruff Issues (11, post auto-fix of 82)

All minor:
- `splits.py`: 2× `int(round(...))` → remove unnecessary int (SIM115)
- `test_synthetic.py`: unused unpacked var `t` → prefix with `_` (RUF059)
- Several more context manager suggestions (SIM115)
- None in newly added diffusion files

## 8. Open GitHub Issues

To be enumerated in next task via `gh issue list`. Not known at audit time.

## 9. Gaps and Blockers

| # | Gap | Blocked by |
|---|---|---|
| G1a | Canonical independent storm grouping | HUMAN SIGNOFF REQUIRED |
| G1b | License/redistribution/per-station datum audit | HUMAN SIGNOFF REQUIRED |
| G1c | HK boundary event (pearl_river…01149 crosses cut 78904) | HUMAN SIGNOFF REQUIRED |
| G2a | Frozen GRU/TCN baselines on matched 72h→24h | Not yet run |
| G2b | True training A1 oversampling (not prediction-noise) | Not yet implemented |
| G3a | DDPM root cause diagnosis (schedule/objective/architecture) | Needs investigation |
| G3b | Controlled retraining of fixed DDPM | G3a + GPU authorization |
| G4 | A0–A4 augmentation utility comparison | G2a + G3b |
| G5 | Full paper | G1–G4 evidence |

## 10. Audit Commitment

All statements here are verified by actual terminal commands executed at audit time. Historical report excerpts are clearly labeled and not claimed as fresh reruns.
