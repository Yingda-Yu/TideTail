# A3 failure mechanism study — BLOCKED, no causal conclusion

Issue #5 prerequisite audit at source `86169e1`; see
[the event audit](../audits/WAVE17_INDEPENDENCE_AUDIT.md).
Completed downstream forecasting runs are preserved. No matched synthetic target
archive was found in the inspected results/predictions inventory. Generating new
samples would be a new diagnostic experiment, not recovery of the exact training
samples. Host job availability cannot be verified here and no GPU was used.

The following diagnostics are required on identical train-only conditioning
windows, with fixed generator seed, checkpoint hash, split hash and station IDs.
All are currently UNMEASURED for corrected v2 matched synthetic targets:

| Diagnostic | Required comparison |
|---|---|
| Mean and standard deviation | Synthetic/real residuals in metres |
| Distribution distance | KS and Wasserstein, uncertainty by storm |
| Positive exceedance | Fixed train-derived q95/q99 |
| Positive quantiles | q95/q99 magnitude errors |
| Peak magnitude | Per-event, equal-event weighting |
| Duration | Consecutive exceedance hours |
| Rise rate | Maximum hourly positive increment |
| Fall rate | Maximum hourly negative increment |
| Temporal correlation | ACF at declared lags |
| Spectrum | Declared frequency-band power |
| Conditional consistency | History end to generated onset; conditional peak errors |
| Diversity | Exact duplicates and scale-aware nearest-neighbor distances |
| Mode collapse | Within-condition replicate variance versus between-condition variance |
| Ratio and region dependence | Full fixed 10/25/50% grid, both seeds/architectures |

Every result needs denominator, independent storm n, and storm-cluster CI or an
explicit insufficient-events statement. A near-duplicate ratio alone cannot
establish collapse without its distance threshold, normalization, reference
population and checkpoint provenance. Historical double-detided A3 remains an
invalid-target engineering control, not evidence about corrected surge physics.

The audit JSON reports all 120 descriptive downstream A3-minus-A0 effects, with
no test-selected “best” claim. Equal optimizer steps are documented, but seed
initialization and independent-event grouping remain unresolved. Distribution
mismatch, conditional mismatch and optimization interference cannot yet be
distinguished. Mechanism acceptance is incomplete; W18/W19/W20 dependent work
remains blocked. Next task: freeze and validate prediction-to-canonical-storm
mapping, then compute event-paired effects before mechanism or model expansion.
