# Wave 3.2.1 — Final Integrity Patch: Revision Report

**Date:** 2026-11-28 · **Scope:** integrity corrections only; experimental
scope CLOSED; no new training, no dataset expansion, no Wave 4.0.

## 1. Bonferroni double correction — FIXED

- **Bug:** the Wave 3.2 "correction" (v2) compared the *adjusted* Bonferroni
  p (`min(1, 10·raw)`) against α/n = 0.005 — a double correction. The two
  correct, equivalent forms are `raw p < 0.005` and `adjusted p < 0.05`.
- **Fix:** `scripts/wave30_arch_interaction.py` v3 uses the raw-p rule;
  regression tests pin rule equivalence and the v2 bug
  (`tests/test_wave32_bonferroni_rule.py`, 4 tests).
- **Corrected result (from committed raw p-values, unchanged):** **4/10
  survivors** — US Gulf standard/tail-aware (raw p = 1e-4), Southeast
  Australia standard (raw p = 4e-4) and **tail-aware (raw p = 8e-4,
  adjusted p = 0.008)**. 8/10 negative interaction directions retained
  (Charleston +0.76/+0.82 mm; NY Bight ns; Thames −2.41/−2.35 mm ns).
- **Synchronised:** abstract, intro contribution, Results (§6.8 text +
  F8 caption), Discussion, Conclusion, F8 (regenerated from v3 JSON),
  CLAIMS_LEDGER (E32 superseded → E39), WAVE32_FINAL_STATISTICAL_AUDIT.md
  (§1, §2 table, §3 rewritten as v1→v2→v3 history, §9 verdicts),
  WAVE32_FINAL_ACCEPTANCE.md, LATEST_PROGRESS.md, WAVE30_TASK_QUEUE.json,
  WAVE30_MASTER_STATUS.md.
- **Other multiple-testing families audited for the same error:** five-region
  primary (raw p < α/20 ✓), physics six-arm (raw p < α/20 ✓),
  mechanism-light (Holm-adjusted p < 0.05 ✓). **No double correction
  elsewhere.**

## 2. Statistical wording corrections

1. **MC standard error:** at p ≈ 1e-4 with 10,000 draws,
   se = sqrt(p(1−p)/n) ≈ **1e-4** — the same order as p itself (the earlier
   "se ≈ 1e-5" was wrong). Audit §1 rewritten; decisions unaffected
   (survivors at raw p ≤ 4e-4; physics family uses exact enumeration).
   Regression test added for the SE floor.
2. **Checkpoint vs scientific acceptance:** §6.2, benchmark doc §2 and the
   acceptance matrix now state explicitly that 70/70 validator success is an
   *artifact/protocol acceptance*, distinct from the per-cell scientific
   gate statuses (REJECTED/INCONCLUSIVE), which are unchanged.
3. **Tail-aware background rejection:** blanket phrasing already carried the
   exception (US Gulf seed 99 autumn background: scale 2.002, q95r 4.898);
   verified in §6.2, benchmark doc and failure analysis — no unqualified
   "never rejected" statement remains.
4. **P3 classification:** now described everywhere as a *validation-frozen
   retrospective exploratory study*; "confirmed prospectively" replaced in
   §6.11 and ledger E38.
5. **Arms vs deployment policy:** §6.11 now states that all candidate arms
   are evaluated and reported individually and that no validation-selected
   deployment policy is chosen or endorsed on test data.
6. **Necessary/sufficient/causal claims:** "Consuming the ignored memory was
   necessary" scoped (no ablation isolating necessity was run);
   "neither necessary nor sufficient" statements scoped with "In our
   data / In our experiments"; Discussion "Causality" limitation rewritten as
   a history-only covariate statement with no causal claim.
7. **Benchmark naming:** described as **seven generator configurations,
   including two Flow Matching variants** (abstract, intro, related work,
   results headings/captions, discussion, benchmark doc); the Diffusion-TS
   target-semantics limitation (trained on observed total water level,
   scored against its own target, not ranked against residual-trained
   families) is retained unchanged.

## 3. Artifact reconciliation

- **CI-tested commit `f63a92d` vs final pushed `ffdaf79`:** the three
  intermediate commits (`3806c22`, `3403d2e`, `ffdaf79`) contained only
  lint-clean edits to the two Wave 3.2 scripts, byte-level figure re-renders
  from unchanged artifacts, build artifacts, CI/acceptance records — and,
  accidentally, two debug-residue files (`scripts/_cds_auth_test.py`,
  `scripts/_launch_gpu_jobs.sh`) which have been **removed from the index**
  (owner constraint; kept untracked on disk, commit `c99a4eb`).
- The full self-hosted CI suite was **re-run after the integrity patch** on
  the final tree (see `data/logs/ci_run_20261128_321.log`).
- Status documents (LATEST_PROGRESS, task queue, master status) updated to
  the v3 state.
- Final manuscript recompiled and all pages inspected.

## 4. Final CI record (post-patch)

| Item | Value |
|---|---|
| Commit | this commit (see git log; recorded in log header) |
| pytest | 338 passed (334 + 4 new Bonferroni-rule regression tests) |
| ruff (gate: src + Wave 3.2/3.2.1 scripts) | clean |
| LaTeX | 40 pages, 0 undefined citations/references, placeholder scan CLEAN |
| Log | `data/logs/ci_run_20261128_321.log` |

## 5. Verdict

**READY FOR OWNER REVIEW** — all integrity corrections passed. Owner-only
items unchanged: author order/affiliations, journal choice, repository
DOI, funding/COI wording. PR #23 remains OPEN and DRAFT; main untouched; no
further experiments launched.
