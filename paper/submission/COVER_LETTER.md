# Cover Letter Draft — TideTail Wave 3.2

Status: DRAFT FOR OWNER REVIEW. Bracketed fields must be filled/confirmed by
the Owner before any submission. No submission will be sent automatically.

---

Dear Editors,

We submit our manuscript entitled "When Does Generative Augmentation Help
Extreme Coastal Water-Level Forecasting? Architecture Dependence, Physical
Priors, and Independent-Event Validation" for consideration in [JOURNAL].

Generative augmentation — training a conditional generative model on a
gauge's own history and adding its synthetic windows to the training set —
is an increasingly popular answer to the rarity of extreme coastal events.
Reported successes, however, are usually aggregate, single-region, and
single-seed. We asked the narrower question in our title, and answered it
with one frozen protocol: five contrasting coastal regions, two downstream
architectures (GRU and TCN), seven generator families under one fidelity
gate and one matched compute budget, a six-arm physical-prior ablation, and
sign-flip inference keyed to independent event units rather than overlapping
windows.

The central finding is deliberately two-sided. Under Bonferroni-corrected
event-level testing, 12 of 20 primary augmentation comparisons survive: six
are genuinely beneficial (all four cells in Southeast Australia, the Gulf
Coast TCN pair) and six genuinely harmful (all four cells in
Thames/North Sea, the Charleston TCN pair). Generator quality measured on
the bulk distribution does not predict this outcome: every generator is
fidelity-INCONCLUSIVE on tail agreement, yet augmentation both improves and
degrades forecasts depending on region. Where augmentation helps, the TCN
extracts a significantly larger benefit than the GRU; where it hurts, the
architecture difference is insignificant. Physics-inspired statistical
regularisation of the generator produced no benefit — the only
Bonferroni-surviving contrasts are increases in error on Charleston GRU.

We highlight three practices that we believe merit the journal's standards:

1. An independent-event census (679 units, 2015--2024) with a frozen
   one-to-one POT/TC matching rule, a documented sensitivity analysis, and a
   correction of a 707-vs-679 double count in our own earlier record.
2. Multiple-testing families declared before evaluation, with Monte Carlo
   resolution, seed uncertainty, and event-unit dependence reported
   explicitly in a statistical audit.
3. Every number traceable to a checked artifact and a checked-in plotting
   script; all known failures, withdrawn pilot results, and
   INCONCLUSIVE strata (extreme events, n=4--5 per region) are retained in
   the manuscript rather than tuned away.

We confirm that this manuscript is original, is not under consideration
elsewhere, and that all authors have approved the submission. The 2022--2024
test period is disclosed in the manuscript as retrospective rather than an
untouched confirmatory holdout; we ask that this framing be preserved in
editorial processing.

Suggested reviewers: [OWNER TO PROVIDE OR OMIT].

Conflicts of interest: [OWNER TO CONFIRM — none declared in the draft
manuscript].

Data and code availability: all training scripts, evaluation code, run
manifests, immutable result artifacts and the plotting scripts for every
figure are in the public repository [OWNER TO CONFIRM REPOSITORY URL FOR
SUBMISSION].

Sincerely,
[Corresponding author — OWNER TO CONFIRM]
on behalf of all authors
