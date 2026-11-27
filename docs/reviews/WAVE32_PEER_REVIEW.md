# Wave 3.2 simulated peer review — FIRST PASS (pre-P2-acceptance manuscript state)

- **Date:** 2026-11-28
- **Reviewer scope:** Independent simulated peer review of `paper/main.tex`, all section files in `paper/sections/` (1-introduction … 8-conclusion, a-audit), `paper/references.bib`, and the compiled `paper/main.pdf` (38 pages, inspected read-only). Manuscript checked against the eight Wave 3.2 ground-truth constraints (GT1–GT8); cross-section numerical consistency, stale-statement scan, statistical-wording audit, and citation-key verification. Read-only review: no files modified except this report; no tests, installs, or interference with running processes.

---

## Summary assessment

The manuscript is unusually strong on statistical hygiene — declared testing families, event-unit pairing, preserved negative results, and honest INCONCLUSIVE labelling — and every headline number (12/20 with 6 beneficial / 6 harmful, 3/10 interaction, 3/20 physics ablation, 679 census, 9/26 pilot harms, fidelity ratios 0.440/0.084/0.181/1.048) is internally consistent across abstract, results, discussion and conclusion. It is nevertheless not yet submittable: one Results subsection is still written in pending/future tense, the seven-family season-stratified benchmark promised in Contribution 2 is never actually reported, the Experimental Setup section describes a retired 8:1:1 split as if it were the operative protocol, and the same standard-FM arm carries contradictory values across three tables from different waves without reconciliation.

---

## MAJOR issues

**M1. A Results section is written in pending tense — the bounded tail-calibration study has not run.**
`paper/sections/6-results.tex:745–747` (PDF p. 27): "*Status: this section is finalised after the bounded run completes; the study is classified as an exploratory extension…*". A journal manuscript cannot contain a section whose content is deferred to a future run; §7.5 and §8 also point readers to it as if it existed.
**Fix:** either (a) complete the bounded run and report the six-arm outcome with the same table discipline as Tables 5–6, or (b) rewrite §6.10 as a short "pre-specified extension protocol" paragraph (no Status sentence), explicitly labelled not-yet-executed, and adjust §7.5/§8 references accordingly. Option (b) is acceptable only if no conclusion depends on it — currently none does.

**M2. Contribution 2 promises a 5 regions × 7 families × 2 seeds season-stratified benchmark that the Results never report.**
`paper/sections/1-introduction.tex:77–86` claims seven families "on identical splits, with season-stratified fidelity scoring, background/moderate/extreme strata, and checkpoint verification … for every cell of the 5×7×2 generator-seed grid"; the abstract (`main.tex:34–36`) and conclusion (`8-conclusion.tex:26–29`) repeat this. The Results contain only (i) a **four**-generator production benchmark (§6.2, Table 2: VAE, TimeGrad, Diffusion-TS, FM × 5 regions × 2 seeds) and (ii) a **two-region, one-seed** seven-family *pilot* whose fidelity was computed on the first 200 consecutive test windows (§6.4), explicitly superseded for fidelity statements. The season-stratified seven-family table does not exist in the paper, although §5.6 states the refresh "was run".
**Fix:** add the season-stratified seven-family benchmark table (the data reportedly exists), or rescope Contribution 2, the abstract, §7.7 and the conclusion to what is actually shown: a 4-family stratified production benchmark plus a 7-family two-region fixed-budget pilot.

**M3. The same standard-FM augmentation arm carries contradictory numbers across three tables without reconciliation.**
- §6.7 corrected pilot (`6-results.tex:458–462`): Charleston GRU **+0.86%** harm (p=0.016, 40 units); NY Bight GRU **+0.92%** harm (p=0.0023, 51 units).
- §6.9 Wave 2.8 physics-ablation table (`6-results.tex:696–699`), "std" column: Charleston GRU **+0.70**, NY Bight GRU **−1.35**, Charleston TCN **−0.32**, NY Bight TCN **+0.07**.
- §6.8 Wave 3.0 five-region table (`6-results.tex:566–580`): Charleston GRU **+0.89\***, NY Bight GRU **−0.18**, Charleston TCN **+1.56\***, NY Bight TCN **−0.70**.
Charleston TCN flips sign between Table 6 (−0.32, improvement) and Table 5 (+1.56, significant harm); NY Bight GRU flips between §6.7 (+0.92 harm) and Table 5 (−0.18, null). Relatedly, §7.6 (`7-discussion.tex:163–164`) states the physics contrasts use **n = 14–15 units** (matching the 2024 extreme-unit column of Table 1) while Tables 5 uses 40/51 POT units for the same regions — the unit basis of each table is never stated side by side.
**Fix:** add an explicit reconciliation paragraph (or table footnote) stating which test unit set and which run batch each table uses (2022–2024 POT units vs 2024 extreme units; generator/downstream seed sets), and why the Wave 2.8 and Wave 3.0 std-arm estimates differ (seed variance is the paper's own explanation — cite the NY-Bight seed-flip observation). Without this, a reviewer will read Tables 5 and 6 as contradictory.

**M4. §5.2 "Split protocol" describes the retired 8:1:1 split as the operative protocol.**
`paper/sections/5-experiments.tex:24–37` presents "Event-grouped chronological 8:1:1 split" with full-support embargo as *the* split protocol, with no retrospective-test disclosure. This contradicts §3.4 (`3-protocol.tex:77–103`), which freezes 2015–2020/2021/2022–2024 and explicitly relegates the 8:1:1/2022-only protocol to the historical candidate zoo, and contradicts §6.7's window counts (52,441/8,593/26,137), which are calendar-split counts, not 8:1:1 counts.
**Fix:** rewrite §5.2 to describe the frozen calendar split (event-grouped, 168 h embargo, train-only scalers/thresholds) and add the one-sentence retrospective disclosure that §3.4 already carries; mention the 8:1:1 split only as the historical-zoo protocol.

---

## MINOR issues

**m1. §7.1 opening sentence overstates the fidelity-gate status of Flow Matching.**
`7-discussion.tex:7–8`: "Flow Matching is the only generator that **passes the fidelity gate** in every wave", while §6.2 (`6-results.tex:117–119`) and §7.1 itself state every FM run is formally *inconclusive*, never accepted. Replace with "is the only generator never rejected by the gate / the only fidelity-eligible generator".

**m2. Historical candidate-zoo aggregates presented in current-tense general claims.**
`6-results.tex:311–322` ("Architecture dependence"/"Region dependence", Table 4, Fig. `W27_architecture_region`): "the TCN responds to augmentation roughly forty times more strongly" derives from the historical 320-test family and sits in tension with the current 3/10 interaction finding. The scope note at `6-results.tex:282–289` is good; extend the same "(historical candidate zoo)" tag to these two paragraphs, Table 4's caption, and Fig. `fig:arch`.

**m3. Stale cross-reference: "six-cell follow-up … is reported in the Wave 2.8 results".**
`6-results.tex:468–470`. The GRU/TCN × std/tail follow-up is the Wave 3.0 five-region table (§6.8); Wave 2.8 is the physics ablation, which does not contain the plain tail-aware arm. Point the reader to Table 5 / §6.8.

**m4. Unexplained window counts in §6.1.**
`6-results.tex:61–64`: "8,601 test windows in four regions and 6,968 in Thames (24.7% NaN gaps)". Neither 8,601 nor 6,968 matches any count of the frozen protocol (26,137 test windows 2022–2024; 8,593 for 2021), and 6,968 ≈ 81% of 8,601 does not follow trivially from a 24.7% NaN rate. State which protocol/period these audit windows belong to and how NaN gaps map to window counts.

**m5. Parameter-count claims in §4.5 do not match the DDPM actually used.**
`4-method.tex:112–114` says the main-task 96→72 DDPM parameter count "remains in the 30–40K range", but the conditional DDPM benchmarked in §6.4 has 27,937 parameters (`6-results.tex:225`); likewise "GRU hidden ≈5–10K matched to DDPM parameter budget" (`4-method.tex:155–156`) is loose. Align the numbers or soften "matched".

**m6. Checkpoint-verification claim has no results or audit backing.**
Abstract (`main.tex:35–36`) and §5.6 (`5-experiments.tex:132–137`) claim every generative checkpoint is verified (tensor counts, generator-only vs auxiliary params, SHA256), but §6.2 reports no verification outcome and the appendix evidence table (`a-audit.tex:21–49`) has no row for it. Add a verification summary sentence to §6.2 and an evidence-table entry.

**m7. Figure caption scope error for the representative-extremes figure.**
`6-results.tex:336–337` captions `W32_representative_extremes` as "historical candidate-zoo archives (retrospective **2022–2024** test window)", but the candidate zoo's test year was 2022-only (§3.4 `3-protocol.tex:99–102`; §6.5 scope note). Correct the parenthetical.

**m8. Internal repository detail in a journal manuscript.**
Absolute dataset paths (`3-protocol.tex:5`), issue numbers ("Issue #6", `4-method.tex:139`, `5-experiments.tex:148`), branch/PR state (`a-audit.tex:10–12`), and GPU UUIDs (`5-experiments.tex:149–150`) are inappropriate for an SCI submission body. Move to a supplement/reproducibility annex or anonymise; keep the audit content but reframe for external readers.

---

## Compliance checklist against the 8 ground truths

| # | Ground truth | Verdict | Evidence |
|---|---|---|---|
| 1 | Five-region primary: 12/20 Bonferroni survivors at α=0.0025, 6 beneficial / 6 harmful; SE-Aus all 4 improve; Thames all 4 harm; Gulf TCN improves; Charleston TCN harms; NY Bight null-to-mixed | **PASS** | Table 5 (`6-results.tex:566–580`): starred cells = Charleston 2 harm + Gulf 2 benefit + SE-Aus 4 benefit + Thames 4 harm = 12 (6/6); NY Bight TCN −0.71 at p=0.0095 correctly reported below threshold (`6-results.tex:492–493`); ranges −1.29…−3.02 / +3.44…+4.31 identical in abstract, intro, results, conclusion. |
| 2 | Architecture interaction: 3/10 survive α=0.005 (Gulf std/tail, SE-Aus std); SE-Aus tail p_bonf=0.008 does NOT survive; never 4/10; 8/10 negative | **PASS** | `6-results.tex:497–514` (3 survivors; tail-aware "falls just short", p_bonf=0.008; "8 of 10 interactions are negative"); Fig. caption `6-results.tex:545–553`; §7.3 `7-discussion.tex:81–90`; abstract & intro say 3/10. Grep confirms no "4 of 10"/"4/10" anywhere. |
| 3 | Physics ablation: 3/20 survivors, all Charleston GRU, all increased error; priors = ERA5 conditioning + physics-inspired statistical regularisation, never PDE constraints | **PASS** | Table 6 (`6-results.tex:702–708`): only C2/C3/C4 Charleston GRU flagged †, all +0.55…+0.57 mm (harm); footnote `6-results.tex:708`. PDE disclaimers present in abstract (`main.tex:55–58`), §2.5 (`2-related-work.tex:84–88`), §4.4 (`4-method.tex:85–89`), §6.9 (`6-results.tex:675–678`), §7.7, §8. All 11 PDE/bathymetry mentions are disclaimers. |
| 4 | 2022–2024 test is retrospective; every downstream claim carries the disclosure | **PASS** | Title footnote (`main.tex:17`), §1.2 (`1-introduction.tex:126–128`), §3.4 (`3-protocol.tex:92–97`), §6.7 (`6-results.tex:465–467`), §6.8 (`6-results.tex:528–529` "All conclusions are retrospective"), §6.10, §7.8 first limitation, §8. (Caveat: §5.2 lacks it — addressed as M4.) |
| 5 | Extreme stratum (n=4–5/region) labelled INCONCLUSIVE; INCONCLUSIVE models never called validated | **PASS (one wording flag → m1)** | §4.3 (`4-method.tex:69–73`) and §7.8 (`7-discussion.tex:233–236`): all extreme-stratum results INCONCLUSIVE by design, never pooled; Table 5/6 captions "All generators are fidelity-INCONCLUSIVE". Exception: §7.1 "passes the fidelity gate" (m1). |
| 6 | 180 five-region event units are NOT 180 globally independent storms; no cross-region pooling claims | **PASS** | §7.6 (`7-discussion.tex:169–174`): "units are not storms … every p-value is a within-region paired statement, and regional heterogeneity is tested on region-level estimates, never on pooled units"; §1 contribution 1 (`1-introduction.tex:75–76`) "never pooled as independent storms". Unit total 40+51+37+20+32 = 180 ✓. |
| 7 | H1 not claimed as universally supported | **PASS** | §1.2 (`1-introduction.tex:124–128`): "We do not claim H1 is universally supported…"; abstract frames "region- and architecture-dependent intervention"; conclusion "not 'yes' or 'no'". |
| 8 | Bibliography: only verified entries; every \cite key exists in references.bib | **PASS** | 40 unique cited keys in sections/main.tex; all 40 present in `paper/references.bib` (1:1, no orphans, no missing); per-entry VERIFIED comments with source checks (e.g. Diffusion-TS correction to yuan2024diffusionts, TimeGrad arXiv-ID correction); compiled PDF has zero "??"/unresolved citations (38 pages, verified via read-only inspection). |

---

## Verdict (READY / NEEDS FIX per stop-criterion aspect)

| Aspect | Verdict | Note |
|---|---|---|
| 1. Five-region primary claims (12/20, 6/6, region pattern) | **READY** | Fully consistent across abstract/intro/results/discussion/conclusion. |
| 2. Architecture-interaction claims (3/10, SE-Aus tail non-survivor, 8/10 negative) | **READY** | Correct everywhere; historical-zoo framing needs a tag only (m2). |
| 3. Physics-ablation claims & PDE framing (3/20, Charleston GRU, harmful; no PDE implication) | **READY** | Unit-basis disclosure advisable (part of M3). |
| 4. Retrospective-test disclosure on all downstream claims | **READY** | Disclosure present at every downstream claim site; fix §5.2 protocol description (M4) alongside. |
| 5. Extreme-stratum INCONCLUSIVE discipline / fidelity-gate wording | **NEEDS FIX** | One sentence in §7.1 contradicts the paper's own INCONCLUSIVE rule (m1); trivial fix. |
| 6. Event-unit independence caveats (180 ≠ 180 independent storms) | **READY** | Correctly stated and enforced in both inference and heterogeneity testing. |
| 7. H1 scoping (no universal-support claim) | **READY** | Explicitly bounded in §1.2 and honoured throughout. |
| 8. Bibliography integrity (verified entries, keys exist) | **READY** | 40/40 keys resolve; verification trail embedded in the .bib. |

**Overall: NEEDS FIX (major revision before P2 acceptance).** The blocking items are M1 (pending-tense Results section), M2 (unreported promised benchmark), M3 (unreconciled cross-wave contradictions for the same arm), and M4 (retired split described as operative). All four are editorial/consistency fixes, not new experiments (except the optional completion of the bounded tail-calibration run under M1a). No ground-truth violation was found: all eight compliance items pass, with one wording flag (m1).

---

## Author fix log (2026-11-28, post-first-pass)

| Issue | Disposition |
|---|---|
| M1 (pending-tense tail-calibration) | DEFERRED — P3 bounded run executes serially after P2 completes; section will be finalised with real results or rewritten as a pre-specified protocol paragraph with the deferral documented (owner stop-criterion 5 allows either) |
| M2 (promised 5×7×2 stratified benchmark unreported) | DEFERRED BY DESIGN — the running P2 refresh produces exactly this table; Results will gain the stratified seven-family table + F11 at P2 acceptance |
| M3 (cross-wave std-arm contradictions) | FIXED — reconciliation paragraph added after Table 5 (unit bases, run batches, seed-variance explanation, frozen-protocol precedence) |
| M4 (retired 8:1:1 split as operative) | FIXED — §5.2 rewritten: operative frozen calendar split first, 8:1:1 relegated to historical, shared guarantees retained |
| m1 ("passes the fidelity gate") | FIXED — "never REJECTED by the fidelity gate" |
| m2 (historical aggregates in current tense) | FIXED — "(historical candidate zoo)" tags on both paragraphs, Table 4 caption, fig:arch caption, with pointer to the controlled interaction test |
| m3 (stale Wave 2.8 cross-ref) | FIXED — points to Table 5 / §6.8 as superseding |
| m4 (unexplained 8,601/6,968) | FIXED — attributed to the Wave 1.7 historical candidate-zoo audit under the retired split |
| m5 (param-count claims) | FIXED — cites the actual benchmarked 27,937; "matched" softened |
| m6 (checkpoint-verification evidence) | DEFERRED — verification summary + appendix row land at P2 acceptance (strict validator output) |
| m7 (caption test-year scope) | FIXED — caption and figure footnote now say "retired 8:1:1 split, 2022 test year"; F10 regenerated |
| m8 (internal repo detail) | FIXED — absolute path, issue number, GPU UUID and branch/PR references removed/anonymised |

Re-compile after fixes: 38 pages, 0 undefined citations/references (latexmk exit 0).
