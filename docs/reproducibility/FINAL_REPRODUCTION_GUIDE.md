# Final Reproduction Guide — TideTail Wave 3.2

Status: LIVING DOCUMENT — the CI section is updated after the final
self-hosted CI run on the manuscript's final commit. This guide describes
how to reproduce every published number from a clean checkout.

## 1. Environment

- Linux, Python 3.10; CPU-only reproduction is possible for everything
  except the legacy GPU-recomputed legacy three-region inference (see §5).
- Install: `pip install -e .` (src layout, package `tidetail`); test extras:
  `pip install pytest ruff`.
- GPU0 is prohibited by project policy; no step below requires it.

## 2. Tests and lint

```bash
PYTHONPATH=src python3 -m pytest -q          # full suite (see §7 for the last recorded count)
PYTHONPATH=src python3 -m pytest -q tests/test_wave30_checkpoint_validation.py  # validator regressions
ruff check src scripts
```

## 3. Data pipeline (from raw sources)

The raw gauge/reanalysis ingestion is not re-run for reproduction; the
processed per-region series and manifests are checked in under `data/`
(heavy binaries via LFS/force-add where needed). Reproduce the derived
artifacts:

```bash
# Event census (679 independent units; strict POT/TC matching + tolerance sensitivity)
python3 scripts/event_set_audit.py                 # -> data/results/wave27_event_set_audit.json

# ERA5 readiness matrix (integrity check)
python3 scripts/verify_era5_readiness.py           # -> data/results/era5_readiness_matrix.json
```

## 4. Generative benchmark

### 4.1 Historical four-generator production benchmark
```bash
python3 scripts/train_gen_bench.py                 # original production runs (Figure 4 source)
```

### 4.2 Seven-family pilot (one matched budget)
```bash
python3 scripts/wave28_gen_bench.py --regions south_atlantic_charleston ny_bight \
    --out data/results/wave28_gen_benchmark.json
```
(Reproduces `wave28_gen_benchmark.json`; do not overwrite the checked-in
artifact — use a scratch output path and compare.)

### 4.3 Stratified 70-cell benchmark (P2, Wave 3.0/3.2)
```bash
python3 -u scripts/wave28_gen_bench_revised_eval.py \
    --regions south_atlantic_charleston ny_bight us_gulf_coast \
              southeast_australia thames_north_sea \
    --out data/results/wave30_generator_fidelity.json --resume --refresh
# strict acceptance (run only after the refresh process exits):
python3 scripts/wave30_validate_checkpoints.py     # 70/70 gate, SHA256, param audit
python3 scripts/wave30_fidelity_stratified_csv.py  # -> wave30_generator_fidelity_stratified.csv
```

## 5. Downstream forecasting and inference

### 5.1 Five-region expanded evaluation (primary result)
```bash
python3 scripts/wave28_expanded_eval.py --help     # frozen 2015-2020/2021/2022-2024 protocol
# canonical invocation recorded in docs/experiments/WAVE30_FIVE_REGION_EVAL.md
# -> data/results/wave30_expanded_five_region.json + wave30_expanded_event_metrics.csv
```

### 5.2 Architecture interaction (v2, Bonferroni-corrected)
```bash
python3 scripts/wave30_arch_interaction.py         # -> wave30_architecture_interaction.json
```

### 5.3 Mechanism-light probes (heterogeneity, severity, season)
```bash
python3 scripts/wave30_mechanism_light.py          # -> wave30_mechanism_light.json
```

### 5.4 Six-arm physics ablation
```bash
python3 scripts/wave28_physics_fm.py               # -> wave28_physics_fm.json (long CPU run)
```

Note: the legacy three-region recompute used GPU5; the five-region primary
family and all Wave 3.0 inference are CPU-reproducible.

## 6. Figures

Every numerical figure is regenerated from immutable artifacts by a checked-in
script — never edit the PDFs:

```bash
python3 scripts/make_wave18_figures.py   # F1 study regions, F3 decomposition
python3 scripts/make_wave27_figures.py   # F2 census, F4 fidelity gate, F5 recovery, H1-H2
python3 scripts/make_wave28_figures.py   # F6 six-arm ablation, H3
python3 scripts/make_wave32_figures.py   # F7 five-region, F8 interaction, F9 severity,
                                         # F10 representative extremes, F11/F12 (gated)
```

Figure-to-artifact mapping: see `paper/FIGURE_REGISTRY.md`.

## 7. Self-hosted CI record (updated at final acceptance)

The project's acceptance authority is Owner-managed self-hosted CI, not
GitHub-hosted Actions.

| Item | Value |
|---|---|
| Final HEAD | Wave 3.2 final (see `data/logs/ci_run_20261128_final.log` header) |
| Runner identity | GuanchaoTong (self-hosted, CPU-only) |
| Commands | `PYTHONPATH=src python3 -m pytest -q`; `ruff check src scripts`; `latexmk -pdf main.tex` in `paper/` |
| Exit codes | pytest 0 / ruff 0 / latexmk 0 |
| Test count | 334 passed (2026-11-28 final; 320 at `d0b577a` on 2026-11-26) |
| Ruff | src gate clean (legacy scripts/ debt disclosed in log) |
| Log | `data/logs/ci_run_20261126_p5.log`; `data/logs/ci_run_20261128_final.log` |
| LaTeX | 40 pages, 0 undefined refs (final) |

## 8. Manuscript build

```bash
cd paper && latexmk -pdf main.tex     # main.pdf; verify 0 undefined citations/references
```

## 9. Provenance rules

- `data/results/RUN_MANIFEST.jsonl` records script SHA256, workspace state
  and launch configuration for every heavy run.
- Preserved runs and their backups are never overwritten; new runs write
  fresh files or `backups/`.
- Every figure/table in the paper traces to one of the artifacts above; the
  mapping is `paper/FIGURE_REGISTRY.md`, and claim-level traceability is
  `paper/CLAIMS_LEDGER.md` (E18-E36).
