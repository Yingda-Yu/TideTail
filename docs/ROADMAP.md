# Development roadmap

**G0 Repository foundation:** packaging, CPU CI, schemas, toy-data tests, contributor/agent contract, paper outline and claims ledger.

**G1 Data integrity:** legal source inventory; download/checksum manifest; multi-gauge QC, units/time alignment, tide/surge target definition; event-aware chronological train/val/test split and leakage tests. No GPU training until signed off.

**G2 Forecast baseline:** naive and strong GRU/TCN/Transformer baselines; matched train-only oversampling; frozen evaluation metrics and uncertainty.

**G3 Diffusion pilot:** CPU smoke; standard conditional diffusion and tail-aware conditional diffusion; memorize/physics checks; low-budget pilot on real held-out storm events.

**G4 Method study:** tail guidance or weighting, residual generation, multi-gauge lag and forcing consistency; controlled ablations and transfer tests. Proceed only after pilot evidence.

**G5 Paper and release:** reproducible figures, related-work audit, negative findings, final integrity review and artifact release within license limitations.

Parallel execution: data acquisition, literature, package scaffolding, and evaluation API can proceed separately; model training depends on frozen G1 splits. PRs never auto-merge.
