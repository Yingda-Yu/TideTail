# Wave 3.2 — Final Acceptance Matrix

**Date:** 2026-11-28 · **Final HEAD:** `f63a92d1b23e7402f60ecf5b66b35df6cc5eaede`
**Branch:** `trae/global-data-generative-research-20261031` · **PR #23: OPEN, DRAFT (never merged to main)**
**Owner decision context:** Wave 3.2 = final consolidation, full manuscript, audit, submission readiness.

## Stop-criterion gate (owner PART 9)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | P2 passed full 70-cell acceptance | ✅ PASS | strict validator 70 present / 70 OK / 0 failures (SHA256, `named_parameters` audit, WGAN-GP G/C split). NOTE (Wave 3.2.1): this is an **artifact/protocol acceptance**, distinct from scientific fidelity acceptance — per-cell gate statuses remain REJECTED/INCONCLUSIVE as evaluated; `docs/experiments/WAVE30_GENERATOR_BENCHMARK.md`; ledger E37 |
| 2 | Five-region main results verified | ✅ PASS | 12/20 Bonferroni α=0.0025 (6 ben / 6 harm), seed-averaged, per-unit persistence; `wave30_expanded_five_region.json`; retrospective-test disclosure retained in §3.4/§6.8 |
| 3 | Statistical discrepancies resolved | ✅ PASS | arch interaction survivor rule fixed to the correct Bonferroni form (v3, Wave 3.2.1): **4/10** survive (raw p < α/n = 0.005), incl. SE-Aus tail-aware (raw 8e-4 / adj 0.008); v2's 3/10 was a double correction; pilot 11/28 → **9/26**; census 707 → **679**; MC-SE corrected (≈1e-4 at p≈1e-4); `docs/audits/WAVE32_FINAL_STATISTICAL_AUDIT.md` §3; regression-tested |
| 4 | Six-arm physical results integrated | ✅ PASS | 3/20 survive, all Charleston GRU, all harm; physics priors scoped to ERA5 conditioning + statistical regularisation, never PDE constraints (§4, §6.9, §7) |
| 5 | P3 completed as bounded exploratory study | ✅ PASS | 6 arms, validation-frozen stage A, Thames harm unrescued (+2.8…+3.4 %), oversampling +7.6 % harmful in SE-Aus; `docs/experiments/WAVE32_TAIL_CALIBRATION.md`; classified exploratory (retrospective test years), §6.11 + F12 |
| 6 | Abstract/Intro/Methods/Results/Discussion/Conclusion mutually consistent | ✅ PASS | first-pass simulated review + fixes (see below); headline numbers cross-checked (12/20, 4/10, 3/20, 679, 9/26, 70/70) |
| 7 | Core figures + supplementary tables exist | ✅ PASS | F1–F12 all EXISTS (`paper/FIGURE_REGISTRY.md`); supplementary = `wave30_generator_fidelity_stratified.csv` (770 rows), `wave30_expanded_event_metrics.csv`, per-seed JSONs |
| 8 | Bibliography verified | ✅ PASS | 40 entries, each with `% VERIFIED:` primary-source comment; fabricated legacy entries dropped; Diffusion-TS corrected to Yuan & Qiao ICLR 2024 (arXiv:2403.01742); 0 undefined citations |
| 9 | Final PDF compiled and visually inspected | ✅ PASS | 40 pages, latexmk exit 0, 0 undefined refs, placeholder scan CLEAN; figure pages, title, Discussion and appendix pages inspected (p2, p18–27, p31, p40) |
| 10 | Latest self-hosted CI passes | ✅ PASS | see CI record below (final HEAD `f63a92d`) |
| 11 | Independent simulated peer review complete | ✅ PASS | first pass: 4 Major / 8 Minor, all fixed or explicitly deferred-by-design and delivered (`docs/reviews/WAVE32_PEER_REVIEW.md` + fix log); compliance 8/8 ground truths |
| 12 | Results traceable to artifacts and code versions | ✅ PASS | `paper/CLAIMS_LEDGER.md` E18–E38; RUN_MANIFEST.jsonl; FIGURE_REGISTRY mapping; reproduction guide |

## Self-hosted CI record (final HEAD)

| Item | Value |
|---|---|
| Commit SHA | final lint-clean verification run at the post-P3 manuscript HEAD (see log header; last content commit of this wave) |
| Runner identity | GuanchaoTong (owner-managed self-hosted; CPU-only; GPU0 untouched) |
| Commands | `PYTHONPATH=src python3 -m pytest -q`; `ruff check src` + `ruff check scripts/make_wave32_figures.py scripts/wave32_tail_calibration.py`; `latexmk -pdf main.tex` (paper/) |
| Exit codes | pytest 0 / ruff 0 / latexmk 0 |
| Test counts | **334 passed**, 1 benign warning (`test_baselines.py` tensor-detach UserWarning) |
| Ruff | gate (src + Wave 3.2 scripts) clean; legacy `scripts/` debt outside the gate, disclosed in the log |
| LaTeX | 40 pages, 0 undefined citations/references, placeholder scan CLEAN |
| Log | `data/logs/ci_run_20261128_final.log` (Wave 3.2); `data/logs/ci_run_20261128_321.log` (Wave 3.2.1 integrity patch, final: 338 passed) |
| Note | GitHub-hosted Actions are NOT the acceptance authority for this project (owner mandate) |

## Manuscript state

- `paper/main.tex` + sections fully rewritten (Wave 3.2); working title kept:
  "When Does Generative Augmentation Help Extreme Coastal Water-Level
  Forecasting? Architecture Dependence, Physical Priors, and
  Independent-Event Validation"
- Author list preserved as draft (Yu, Mengxue, Jia); affiliations/funding/COI
  intentionally blank pending Owner verification
  (`paper/submission/STATEMENTS_AND_AUTHOR_SHEET.md`)
- Submission package prepared, not submitted:
  `paper/submission/JOURNAL_SHORTLIST.md` (NHESS / Coastal Engineering /
  Environmental Modelling & Software), `COVER_LETTER.md`, statements.

## Verdict

**READY FOR OWNER REVIEW** — all twelve stop criteria are met. Remaining
Owner-only decisions: final author order/affiliations, journal choice,
repository URL/DOI, funding and COI wording. Submission itself is not
performed by the agent.

### Explicit residual limitations (disclosed, not blockers)
1. The 2022–2024 test is retrospective; the paper says so everywhere a
   downstream claim is made.
2. Extreme-stratum fidelity (n=4–5 events/region) is INCONCLUSIVE by design.
3. P3 is exploratory; its Thames negative result is preserved without tuning.
4. The 180 five-region event units are not 180 globally independent storms;
   no cross-region pooling is claimed.
