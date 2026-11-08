# Figure Registry — TideTail

Updated 2026-11-05 Wave 2.7.

All Wave 2.7 figures are generated **only** from committed result artifacts by
`scripts/make_wave27_figures.py`; no value is typed by hand.

| figure_id | title | status | output | artifact source |
|---|---|---|---|---|
| W27-schematic | Pipeline and causal physics boundary | COMPLETE | `paper/figures/W27_schematic.pdf` | conceptual; causal rule guarded by `tests/test_physics_causality.py` |
| W27-census | Decade event census vs union; 2024 units | COMPLETE | `paper/figures/W27_event_census.pdf` | `data/results/wave27_event_set_audit.json` |
| W27-fidelity | Generator scale ratio against gate | COMPLETE | `paper/figures/W27_fidelity_gate.pdf` | `data/results/{vae,timegrad,diffusion_ts,flow_matching}_96_72.json` |
| W27-arch | Augmentation effect by region and architecture | COMPLETE | `paper/figures/W27_architecture_region.pdf` | `data/results/paired_significance_wave27.json` |
| W27-recovery | Legacy vs repaired generator scale | COMPLETE | `paper/figures/W27_recovery.pdf` | `data/results/wave27_model_recovery.json` |
| W27-seedvar | A0 baseline seed variance | COMPLETE | `paper/figures/W27_baseline_seed.pdf` | `data/results/paired_significance_wave27.json` |
| W27-expanded | Expanded-protocol fidelity vs delta | PENDING | — | `data/results/wave27_expanded_evaluation.json` |
| W27-physics | Five-variant physics ablation | PENDING | — | `data/results/wave27_physics_fm.json` |

Notes:

* `W27-expanded` and `W27-physics` are pending because their result artifacts
  either do not yet exist (physics run in flight) or are better presented as
  tables than as figures.
* Every figure above is cited in the manuscript (`paper/sections/`).
* The Wave 1.8 legacy figures (F1, F3, F4, F5, F6, F8, F9) remain on disk but
  are **not cited** by the current manuscript; they encode the superseded
  Wave 1.7 narrative and must not be reintroduced without re-deriving their data.

**Progress**: 6/8 Wave 2.7 figures complete. Two pending artifacts are declared,
not invented.
